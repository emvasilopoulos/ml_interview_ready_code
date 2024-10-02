from collections import namedtuple
import distutils.version
import pathlib
from typing import Any, Dict, List
import graphviz
import torch
import torch.autograd
import warnings

Node = namedtuple("Node", ("name", "inputs", "attr", "op"))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"


def get_fn_name(fn: torch.Node, show_attrs: bool, max_attr_chars: int):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX) :]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = "%-" + str(col1width) + "s: %" + str(col2width) + "s"
    truncate = lambda s: s[: col2width - 3] + "..." if len(s) > col2width else s
    params = "\n".join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + "\n" + sep + "\n" + params


def make_dot(
    model_output: torch.Tensor,
    params: Dict[Any, Any] = None,
    show_attrs: bool = False,
    show_saved: bool = False,
    max_attr_chars: int = 50,
) -> graphviz.Digraph:
    """Produces Graphviz representation of PyTorch autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        model_output: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
        show_attrs: whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """
    if distutils.version.LooseVersion(
        torch.__version__
    ) < distutils.version.LooseVersion("1.9") and (show_attrs or show_saved):
        warnings.warn(
            "make_dot: showing grad_fn attributes and saved variables"
            " requires PyTorch version >= 1.9. (This does NOT apply to"
            " saved tensors saved by custom autograd functions.)"
        )

    if params is not None:
        assert all(isinstance(p, torch.autograd.Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="10",
        ranksep="0.1",
        height="0.2",
        fontname="monospace",
    )
    dot = graphviz.Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ""
        return "%s\n %s" % (name, size_to_str(var.size()))

    def add_nodes(fn: torch.Node):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        if show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX) :]
                if torch.is_tensor(val):
                    dot.edge(str(id(fn)), str(id(val)), dir="none")
                    dot.node(str(id(val)), get_var_name(val, attr), fillcolor="orange")
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + "[%s]" % str(i)
                            dot.edge(str(id(fn)), str(id(t)), dir="none")
                            dot.node(
                                str(id(t)), get_var_name(t, name), fillcolor="orange"
                            )

        if hasattr(fn, "variable"):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            seen.add(var)
            dot.node(str(id(var)), get_var_name(var), fillcolor="lightblue")
            dot.edge(str(id(var)), str(id(fn)))

        # add the node for this grad_fn
        dot.node(str(id(fn)), get_fn_name(fn, show_attrs, max_attr_chars))

        # recurse
        if hasattr(fn, "next_functions"):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        if hasattr(fn, "saved_tensors"):
            for t in fn.saved_tensors:
                seen.add(t)
                dot.edge(str(id(t)), str(id(fn)), dir="none")
                dot.node(str(id(t)), get_var_name(t), fillcolor="orange")

    def add_base_tensor(var: torch.Tensor, color="darkolivegreen1"):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if var.grad_fn:
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color="darkolivegreen3")
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")

    # handle multiple outputs
    if isinstance(model_output, tuple):
        for v in model_output:
            add_base_tensor(v)
    else:
        add_base_tensor(model_output)

    resize_graph(dot)

    return dot


def resize_graph(
    dot: graphviz.Digraph, size_per_element: float = 0.15, min_size: int = 12
):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def save_model_graph_as_png(
    model: torch.nn.Module, example_input: torch.Tensor, save_path: pathlib.Path
):

    output = model(example_input)
    make_dot(output, params=dict(model.named_parameters())).render(
        save_path.as_posix(), format="png"
    )
