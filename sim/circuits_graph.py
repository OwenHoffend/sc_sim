import numpy as np
import copy
import graphviz
from functools import reduce
from enum import IntEnum
from sim.PTM import get_func_mat

class VState(IntEnum):
    UNVISITED      = 0
    FWD_VISIT      = 1 #Node or edge is reached by the selected input nodes 
    BACK_VISIT_1   = 2
    BACK_VISIT_2   = 3
    DUPLICATE      = 4

colors = [
    'gray',
    '#e67639', #orange
    '#3990e6', #light blue
    '#66b830', #greenish
    '#c90c0c'  #red
]

class Edge():
    def __init__(self, src, dest, src_out, dest_in):
        self.src = src
        self.dest = dest
        self.src_out = src_out
        self.dest_in = dest_in
        self.visit_state = VState.UNVISITED

class G_Circuit():
    def __init__(self, n, k, primitive=False, name=''):
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
        self.name = name
        self.ptm = None

        self.subgraph_idx = None #metadata for extracting subgraphs

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

    def render_graphviz(self, fn='graphviz_render'):
        g = graphviz.Digraph(format='png', filename=fn)
        for idx, node in enumerate(self.nodes):
            fillcolor = colors[int(node.visit_state)]
            g.node(str(idx) + " {}".format(node.name), style='filled', fillcolor=fillcolor)

        for edge_list in self.fwd_edges:
            for edge in edge_list:
                edgecolor = colors[int(edge.visit_state)]
                g.edge(str(edge.src) + " {}".format(self.nodes[edge.src].name), \
                    str(edge.dest) + " {}".format(self.nodes[edge.dest].name), color=edgecolor)
        g.view()

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

    def get_edges(self):
        e = []
        for outs in self.fwd_edges:
            e += outs
        return e

    def is_graph_complete(self):
        """Checks that all nodes in the graph have the requried number of ingoing and outgoing edges
        mostly just used as a sanity check"""
        for node in self.nodes:
            if node.connected_n != node.n and not isinstance(node, G_Circuit_IN):
                return False
            if node.connected_k != node.k:
                return False
        return True

    def is_flat(self):
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

    def reset_inputs(self):
        for node in self.nodes:
            node.reset_inputs()

    def eval(self, *args):
        assert self.is_flat() #FIXME: eval is currently broken for non-flat circuits
        assert len(args) == self.n
        self.reset_inputs()
        assert not self.has_all_inputs()
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
                if type(node_outs) == np.ndarray: 
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
        return np.array(outs)

    def fwd_logic_cone(self, selected_inputs):
        """Forward pass of the sub-circuit PTM optimization algorithm from the wk_5_11_22 slides"""

        #Identify the subgraph 
        input_idxs = self.get_input_nodes()
        queue = [input_idxs[i] for i in selected_inputs]
        found_edges = [] #May be unnecessary
        while queue != []:
            node_idx = queue.pop(0)
            node = self.nodes[node_idx]
            node.visit_state = VState.FWD_VISIT
            for edge in self.fwd_edges[node_idx]:
                edge.visit_state = VState.FWD_VISIT
                found_edges.append(edge) #May need to copy edge instance
                dest_idx = edge.dest
                for incoming_edge in self.back_edges[dest_idx]:
                    if incoming_edge.visit_state != VState.FWD_VISIT:
                        break
                else:
                    queue.append(dest_idx)

        #Get the new endpoint edges
        endpoints = []
        for edge in found_edges:
            dest_node = self.nodes[edge.dest]
            if dest_node.visit_state != VState.FWD_VISIT or \
            isinstance(self.nodes[edge.dest], G_Circuit_OUT):
                endpoints.append(edge)
        return endpoints

    def back_logic_cone_phase1(self, selected_edges, n):
        """Backward pass of the sub-circuit PTM optimization algorithm from the wk_5_11_22 slides"""
        G2 = G_Circuit(n, 2)
        G2.add_node(G_Circuit_OUT())
        G2.add_node(G_Circuit_OUT())
        new_edges = []
        queue = []
        subgraph_idx = 2
        for idx, edge in enumerate(selected_edges):
            src_node = self.nodes[edge.src]
            if src_node.visit_state == VState.FWD_VISIT:
                queue.append(edge.src)
                src_node.subgraph_idx = subgraph_idx
                subgraph_idx += 1
                new_edges.append(Edge(src_node.subgraph_idx, idx, edge.src_out, 0))
                edge.visit_state = VState.BACK_VISIT_1

        #Find the relevant nodes/edges
        while queue != []:
            node_idx = queue.pop(0)
            node = self.nodes[node_idx]
            G2.add_node(node)
            node.visit_state = VState.BACK_VISIT_1
            for edge in self.back_edges[node_idx]:
                if edge.visit_state == VState.FWD_VISIT:
                    src_node = self.nodes[edge.src]
                    if src_node.visit_state == VState.FWD_VISIT:
                        queue.append(edge.src)
                        src_node.subgraph_idx = subgraph_idx
                        subgraph_idx += 1
                    assert src_node.subgraph_idx != None
                    assert node.subgraph_idx != None
                    new_edges.append(Edge(src_node.subgraph_idx, node.subgraph_idx, edge.src_out, edge.dest_in))
                    edge.visit_state = VState.BACK_VISIT_1
        for edge in new_edges:
            G2.add_edge(edge.src, edge.src_out, edge.dest, edge.dest_in)
        for edge_list in G2.fwd_edges:
            for edge in edge_list:
                edge.visit_state = VState.BACK_VISIT_1
        for node in G2.nodes:
            node.visit_state = VState.BACK_VISIT_1
        return G2

    def back_logic_cone_phase2(self, selected_edges):
        queue = []
        for edge in selected_edges:
            edge.visit_state = VState.BACK_VISIT_2
            queue.append(edge.src)
        while queue != []:
            node_idx = queue.pop(0)
            node = self.nodes[node_idx]
            if node.visit_state == VState.FWD_VISIT:
                node.visit_state = VState.BACK_VISIT_2
            elif not isinstance(node, G_Circuit_IN):
                node.visit_state = VState.DUPLICATE

            for edge in self.back_edges[node_idx]:
                if edge.visit_state == VState.FWD_VISIT:
                    edge.visit_state = VState.BACK_VISIT_2
                else:
                    edge.visit_state = VState.DUPLICATE
                queue.append(edge.src)

    def get_flattened(self):
        """Convert to a graph representing the connections between primitive nodes only, no sub-circuits"""
        new_graph = copy.deepcopy(self)
        if self.is_flat():
            return new_graph
        
        #identify nodes to flatten
        to_flatten = []
        to_flatten_idx = []
        to_keep = []
        for idx, node in enumerate(self.nodes):
            if node.primitive:
                to_keep.append(node)
            else:
                to_flatten.append(node)
                to_flatten_idx.append(idx)

        for (idx, node) in zip(to_flatten_idx, to_flatten):
            node = node.get_flattened() #Recursive call
            offset = len(new_graph.nodes)

            #insert new nodes
            for sub_node in node.nodes:
                new_graph.add_node(sub_node)

            #break edges that cross the sub-circuit boundary
            for edge in list(new_graph.back_edges[idx]):
                new_graph.add_edge(edge.src, edge.src_out, offset + edge.dest_in, 0)
                new_graph.fwd_edges[edge.src].remove(edge)
                new_graph.back_edges[edge.dest].remove(edge)
            for edge in list(new_graph.fwd_edges[idx]):
                new_graph.add_edge(offset + edge.dest_in, 0, edge.dest, edge.dest_in)
                new_graph.fwd_edges[edge.src].remove(edge)
                new_graph.back_edges[edge.dest].remove(edge)

            #add new edges
            for sub_edge in node.get_edges():
                new_graph.add_edge(sub_edge.src + offset, sub_edge.src_out, sub_edge.dest + offset, sub_edge.dest_in)
        return new_graph

    def get_ptm(self):
        if self.ptm is not None:
            return self.ptm

        #assert self.is_flat()
        #queue = self.get_input_nodes()
        #for node_idx in queue: #set starting ptms
        #    self.nodes[node_idx].get_ptm()
        #while queue != []:
        #    node_idx = queue.pop(0)
        #    for edge in self.fwd_edges[node_idx]:
        #        dest_idx = edge.dest
        #        dest_node = self.nodes[dest_idx]
        #        if dest_node.ptm is not None:
        #            continue
        #        par_ptms = []
        #        for incoming_edge in self.back_edges[dest_idx]:
        #            incoming_ptm = self.nodes[incoming_edge.src].ptm
        #            if incoming_ptm is None:
        #                break
        #            par_ptms.append(incoming_ptm)
        #        else:
        #            par = reduce(np.kron, par_ptms)
        #            dest_node.ptm = par @ dest_node.get_ptm()
        #            queue.append(dest_idx)
        #self.ptm = reduce(np.kron, [self.nodes[out].ptm for out in self.get_output_nodes()])

        def eval_wrapper(*args):
            return self.eval(*args)

        self.ptm = get_func_mat(eval_wrapper, self.n, self.k)
        return self.ptm

class G_Circuit_Prim(G_Circuit):
    def __init__(self, n, k, func, name):
        self.inputs = [None for _ in range(n)]
        self.func = func
        super().__init__(n, k, primitive=True, name=name)

    def add_input(self, val, idx):
        assert not self.has_all_inputs()
        assert idx < self.n
        self.inputs[idx] = val

    def reset_inputs(self):
        self.inputs = [None for _ in range(self.n)]

    def has_all_inputs(self):
        for input in self.inputs:
            if input is None:
                return False
        return True

    def eval(self):
        assert self.has_all_inputs()
        f = self.func(*self.inputs)
        self.reset_inputs()
        return f

    def get_ptm(self):
        if self.ptm is None:
            self.ptm = get_func_mat(self.func, self.n, self.k)
        return self.ptm
        
class G_Circuit_IN(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(1, k, lambda x: np.array([x for _ in range(k)]), 'IN')

    def get_ptm(self): 
        #Inputs are primitive but they can have multiple outputs, however the PTM must be given
        #as though there's only a single output
        if self.ptm is None:
            self.ptm = np.array([
                [True, False],
                [False, True]
            ])
        return self.ptm

class G_Circuit_OUT(G_Circuit_Prim):
    def __init__(self):
        super().__init__(1, 0, lambda x: x, 'OUT')

    def get_ptm(self):
        if self.ptm is None:
            self.ptm = np.array([
                [True, False],
                [False, True]
            ])
        return self.ptm

class G_Circuit_AND(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(2, k, np.bitwise_and, 'AND')

class G_Circuit_OR(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(2, k, np.bitwise_or, 'OR')

class G_Circuit_NOT(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(1, k, np.bitwise_not, 'NOT')

#Circuit library
class G_Circuit_MUX(G_Circuit):
    def __init__(self):
        super().__init__(3, 1, name='MUX')
        self.add_node(G_Circuit_IN())
        self.add_node(G_Circuit_IN())
        self.add_node(G_Circuit_IN(k=2))
        self.add_node(G_Circuit_NOT())
        self.add_node(G_Circuit_AND())
        self.add_node(G_Circuit_AND())
        self.add_node(G_Circuit_OR())
        self.add_node(G_Circuit_OUT())

        self.add_edge(0,0,4,0)
        self.add_edge(1,0,5,0)
        self.add_edge(2,0,3,0)
        self.add_edge(3,0,4,1)
        self.add_edge(2,1,5,1)
        self.add_edge(4,0,6,0)
        self.add_edge(5,0,6,1)
        self.add_edge(6,0,7,0)

class G_Circuit_MUX_PAIR(G_Circuit):
    def __init__(self):
        super().__init__(5, 2, name='MUX_PAIR')
        for _ in range(4):
            self.add_node(G_Circuit_IN())
        self.add_node(G_Circuit_IN(k=2))
        for _ in range(2):
            self.add_node(G_Circuit_MUX())
        for _ in range(2):
            self.add_node(G_Circuit_OUT())
        
        self.add_edge(0,0,5,0)
        self.add_edge(1,0,5,1)
        self.add_edge(2,0,6,0)
        self.add_edge(3,0,6,1)
        self.add_edge(4,0,5,2)
        self.add_edge(4,1,6,2)
        self.add_edge(5,0,7,0)
        self.add_edge(6,0,8,0)