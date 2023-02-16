parameter integer NUM_CONSTS = 3;
parameter integer NUM_VARS = 2;
parameter integer NUM_OUTPUTS = 1;

parameter integer NUM_INPUTS = NUM_CONSTS + NUM_VARS;
parameter integer NUM_WEIGHTS = 2 ** NUM_VARS;
parameter integer WEIGHT_MATRIX [NUM_OUTPUTS * NUM_WEIGHTS-1 : 0]   = { //<-- have python fill this in automatically
    0, 
    3,
    5, 
    8
}; 