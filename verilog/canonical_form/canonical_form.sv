`include "./sc_sim/verilog/canonical_form/circs/gb4.svh"

module canonical_form(
    input [NUM_CONSTS-1:0] const_inputs,
    input [NUM_VARS-1:0] var_inputs,
    output logic [NUM_OUTPUTS-1:0] outputs
);

    logic [2**NUM_CONSTS-1:0] ohot_const;
    logic [2**NUM_VARS-1:0] ohot_vars;

    always_comb begin
        ohot_const = 1 << const_inputs;
        ohot_vars = 1 << var_inputs;
    end

    //AND-OR Network
    genvar k;
    genvar v;
    generate
    for(k = 0; k < NUM_OUTPUTS; k++) begin
        logic [2**NUM_VARS-1:0] and_outputs;
        for(v = 0; v < 2**NUM_VARS; v++) begin
            localparam WEIGHT = WEIGHT_MATRIX[k*(2**NUM_VARS)+v]; 
            if(WEIGHT == 0)
                assign and_outputs[v] = 1'b0;
            else if(WEIGHT == 2**NUM_CONSTS)
                assign and_outputs[v] = ohot_vars[v];
            else
                assign and_outputs[v] = ohot_vars[v] & |ohot_const[WEIGHT-1:0];
        end
        assign outputs[k] = |and_outputs;
    end
    endgenerate
endmodule