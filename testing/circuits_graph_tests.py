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
    rng = bs.SC_RNG()
    lfsr_sz = 6
    N = 2 ** lfsr_sz
    test_inputs = rng.bs_lfsr_p5_consts(N, 5, lfsr_sz, add_zero_state=True)
    results = G.eval(*list(test_inputs))
    print(bs.bs_mean(results[0], bs_len=N))
    print(bs.bs_mean(results[1], bs_len=N))
    print('hi')