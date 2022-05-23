from sim.circuits_graph import *
import sim.bitstreams as bs

def test_MUX_graph():
    """Create the graph for the parallel MUX from the example in wk_5_11_22 slides"""
    print('started')
    G = G_Circuit(5, 2)
    for _ in range(4): #Data inputs
        G.add_node(G_Circuit_IN())
    G.add_node(G_Circuit_IN(k=3)) #Select input
    G.add_node(G_Circuit_NOT(k=2)) #NOT gate for select
    for _ in range(4): #AND gates for main body
        G.add_node(G_Circuit_AND())
    for _ in range(2): #OR gates for main body
        G.add_node(G_Circuit_OR())
    for _ in range(2):
        G.add_node(G_Circuit_OUT()) #Outputs

    print('nodes added')

    #Data inputs to MUX AND gates
    G.add_edge(0, 0, 6, 0)
    G.add_edge(1, 0, 7, 0)
    G.add_edge(2, 0, 8, 0)
    G.add_edge(3, 0, 9, 0)

    #NOT gate input
    G.add_edge(4, 0, 5, 0)

    #Select inputs to MUX AND gates
    G.add_edge(5, 0, 6, 1)
    G.add_edge(5, 1, 8, 1)
    G.add_edge(4, 1, 7, 1)
    G.add_edge(4, 2, 9, 1)

    #Edges between AND and OR gates
    G.add_edge(6, 0, 10, 0)
    G.add_edge(7, 0, 10, 1)
    G.add_edge(8, 0, 11, 0)
    G.add_edge(9, 0, 11, 1)

    #Edges to output nodes
    G.add_edge(10, 0, 12, 0)
    G.add_edge(11, 0, 13, 0)

    print(G.is_graph_complete())
    print(G.is_primitive())
    G.render_graphviz(fn='1')

    #Experimentation with graph-based logic optimization
    selected_inputs = [0, 2, 4]
    endpoints = G.fwd_logic_cone(selected_inputs)
    G.render_graphviz(fn='2')
    selected_endpoints = [endpoints[3], endpoints[0]]
    G2 = G.back_logic_cone_phase1(selected_endpoints, len(selected_inputs))
    remaining_endpoints = [endpoints[1], endpoints[2]]
    G.render_graphviz(fn='5')
    G2.render_graphviz(fn='3')
    G.back_logic_cone_phase2(remaining_endpoints)
    G.render_graphviz(fn='4')

    rng = bs.SC_RNG()
    lfsr_sz = 6
    N = 2 ** lfsr_sz
    test_inputs = rng.bs_lfsr_p5_consts(N, 5, lfsr_sz, add_zero_state=True)
    results = G.eval(*list(test_inputs))
    print(bs.bs_mean(results[0], bs_len=N))
    print(bs.bs_mean(results[1], bs_len=N))

def test_parallel_const_mul():
    G = G_Circuit(6, 2)
    for _ in range(4):
        G.add_node(G_Circuit_IN(k=2))
    for _ in range(2):
        G.add_node(G_Circuit_IN())
    
    #Top constant (0.0625)
    G.add_node(G_Circuit_AND())
    G.add_node(G_Circuit_AND())
    G.add_node(G_Circuit_AND()) #8
    G.add_edge(0, 0, 6, 0)
    G.add_edge(1, 0, 6, 1)
    G.add_edge(6, 0, 7, 0)
    G.add_edge(2, 0, 7, 1)
    G.add_edge(7, 0, 8, 0)
    G.add_edge(3, 0, 8, 1)

    #Bottom constant
    G.add_node(G_Circuit_OR())
    G.add_node(G_Circuit_OR())
    G.add_node(G_Circuit_OR()) #11
    G.add_edge(0, 1, 9, 0)
    G.add_edge(1, 1, 9, 1)
    G.add_edge(9, 0, 10, 0)
    G.add_edge(2, 1, 10, 1)
    G.add_edge(10, 0, 11, 0)
    G.add_edge(3, 1, 11, 1)

    #AND gate multipliers
    G.add_node(G_Circuit_AND()) #12
    G.add_node(G_Circuit_AND()) #13
    G.add_edge(4, 0, 12, 0)
    G.add_edge(11, 0, 12, 1)
    G.add_edge(5, 0, 13, 0)
    G.add_edge(8, 0, 13, 1)

    G.add_node(G_Circuit_OUT())
    G.add_node(G_Circuit_OUT())
    G.add_edge(12, 0, 14, 0)
    G.add_edge(13, 0, 15, 0)

    print(G.is_graph_complete())
    print(G.is_primitive())
    G.render_graphviz('1')

    selected_inputs = [0, 1, 2, 3, 5]
    endpoints = G.fwd_logic_cone(selected_inputs)
    G.render_graphviz(fn='2')
    selected_endpoints = [endpoints[3], endpoints[0]]
    G2 = G.back_logic_cone_phase1(selected_endpoints, len(selected_inputs))
    remaining_endpoints = [endpoints[1], endpoints[2]]
    G2.render_graphviz(fn='3')
    G.render_graphviz(fn='4')