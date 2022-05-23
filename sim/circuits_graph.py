import numpy as np
import graphviz
from enum import IntEnum

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
                last_visited = True
                for incoming_edge in self.back_edges[dest_idx]:
                    if incoming_edge.visit_state != VState.FWD_VISIT:
                        last_visited = False
                if last_visited:
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

class G_Circuit_Prim(G_Circuit):
    def __init__(self, n, k, func, name):
        self.inputs = [None for _ in range(n)]
        self.func = func
        super().__init__(n, k, primitive=True, name=name)

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
        super().__init__(1, k, lambda x: [x for _ in range(k)], 'IN')

class G_Circuit_OUT(G_Circuit_Prim):
    def __init__(self):
        super().__init__(1, 0, lambda x: x, 'OUT')

class G_Circuit_AND(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(2, k, np.bitwise_and, 'AND')

class G_Circuit_OR(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(2, k, np.bitwise_or, 'OR')

class G_Circuit_NOT(G_Circuit_Prim):
    def __init__(self, k=1):
        super().__init__(1, k, np.bitwise_not, 'NOT')