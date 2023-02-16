`ifndef CIRC_SPEC
`define CIRC_SPEC
localparam integer NUM_CONSTS = 3;
localparam integer NUM_VARS = 2;
localparam integer NUM_OUTPUTS = 2;

localparam integer NUM_INPUTS = NUM_CONSTS + NUM_VARS;
localparam integer NUM_WEIGHTS = 2 ** NUM_VARS;
localparam integer WEIGHT_MATRIX [NUM_OUTPUTS * NUM_WEIGHTS-1 : 0]   = { //<-- have python fill this in automatically
    6, 
    7,
    8, 
    0,
    0,
    8,
    7,
    6
};
`endif