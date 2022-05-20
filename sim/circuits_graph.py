import numpy as np
import queue
from enum import Enum

class VState(Enum):
    UNVISITED    = 1
    FWD_VISIT    = 2 #Node or edge is reached by the selected input nodes 
    BACK_VISIT_1 = 3 #Node or edge is contained in the logic cone of the selected output nodes
    BACK_VISIT_2 = 4 #Node or edge needs to be duplicated

class Edge():
    def __init__(self, src, dest, src_out, dest_in):
        self.src = src
        self.dest = dest
        self.src_out = src_out
        self.dest_in = dest_in
        self.visit_state = VState.UNVISITED

class G_Circuit():
    def __init__(self, n, k, primitive=False):
        self.n = n
        self.k = k
        self.connected_n = 0
        self.connected_k = 0
        self.fwd_edges = []
        self.back_edges = []
        self.nodes = []
        self.node_cnt = 0
        self.visit_state = VState.UNVISITED
        self.primitive = primitive

    def add_node(self, node):
        self.nodes.append(node)
        self.fwd_edges.append([]) #Edges leaving this node
        self.back_edges.append([]) #Edges entering this node
        self.node_cnt += 1

    def add_edge(self, src, src_out, dest, dest_in):
        if src >= self.node_cnt or dest >= self.node_cnt:
            raise ValueError("Cannot add edge - Invalid node index")

        if src_out >= self.nodes[src].k or dest_in >= self.nodes[dest].n:
            raise ValueError("Cannot add edge - Invalid src_out or dest_in index")

        E = Edge(src, dest, src_out, dest_in)
        self.fwd_edges[src].append(E)
        self.back_edges[dest].append(E)
        self.nodes[src].connected_k += 1
        self.nodes[dest].connected_n += 1

    def get_input_nodes(self):
        """Return a list of indices for all nodes in this graph that are input nodes"""
        inputs = []
        for idx, node in enumerate(self.nodes):
            if isinstance(node, G_Circuit_IN):
                inputs.append(idx)
        assert len(inputs) == self.n
        return inputs

    def add_input(self, val, idx):
        assert not self.has_all_inputs()
        input_idx = self.get_input_nodes()[idx]
        input_node = self.nodes[input_idx]
        input_node.add_input(val, 0)
    
    def get_output_nodes(self):
        """Return a list of indices for all nodes in this graph that are output nodes"""
        outputs = []
        for idx, node in enumerate(self.nodes):
            if isinstance(node, G_Circuit_OUT):
                outputs.append(idx)
        assert len(outputs) == self.k
        return outputs

    def is_graph_complete(self):
        """Checks that all nodes in the graph have the requried number of ingoing and outgoing edges
        mostly just used as a sanity check"""
        for node in self.nodes:
            if node.connected_n != node.n and not isinstance(node, G_Circuit_IN):
                return False
            if node.connected_k != node.k:
                return False
        return True

    def is_primitive(self):
        """Checks if this circuit is a graph connecting only AND/OR/NOT circuit elements"""
        for node in self.nodes:
            if not node.primitive:
                return False
        return True

    def has_all_inputs(self):
        """Checks that all input nodes to this circuit have values assigned to them"""
        for input_idx in self.get_input_nodes():
            input_node = self.nodes[input_idx]
            if not input_node.has_all_inputs():
                return False
        return True

    def eval(self, *args):
        assert len(args) == self.n
        for idx, arg in enumerate(args):
            self.add_input(arg, idx)
        assert self.has_all_inputs()

        queue = self.get_input_nodes()
        while queue != []:
            node_idx = queue.pop(0)
            node = self.nodes[node_idx]
            node_outs = node.eval()

            for edge in self.fwd_edges[node_idx]:
                dest_idx = edge.dest
                dest_node = self.nodes[dest_idx]
                if type(node_outs) == list: 
                    val = node_outs[edge.src_out]
                else:
                    val = node_outs
                dest_node.add_input(val, edge.dest_in)
                if not isinstance(dest_node, G_Circuit_OUT) and dest_node.has_all_inputs():
                    queue.append(dest_idx)
        outs = []
        for output_idx in self.get_output_nodes():
            output_node = self.nodes[output_idx]
            assert output_node.has_all_inputs()
            outs.append(output_node.eval())
        return outs

    def fwd_logic_cone(self, input_idxs):
        """Forward pass of the sub-circuit PTM optimization algorithm from the wk_5_11_22 slides"""
        pass

class G_Circuit_Prim(G_Circuit):
    def __init__(self, n, k, func):
        self.inputs = [None for _ in range(n)]
        self.func = func
        super().__init__(n, k, primitive=True)

    def add_input(self, val, idx):
        assert not self.has_all_inputs()
        assert idx < self.n
        self.inputs[idx] = val

    def has_all_inputs(self):
        for input in self.inputs:
            if input is None:
                return False
        return True

    def eval(self):
        assert self.has_all_inputs()
        return self.func(*self.inputs)
        
class G_Circuit_IN(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(1, k, lambda x: [x for _ in range(k)])

class G_Circuit_OUT(G_Circuit_Prim):
    def __init__(self):
        super().__init__(1, 0, lambda x: x)

class G_Circuit_AND(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(2, k, np.bitwise_and)

class G_Circuit_OR(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(2, k, np.bitwise_or)

class G_Circuit_NOT(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(1, k, np.bitwise_not)