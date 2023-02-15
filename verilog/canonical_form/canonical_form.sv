`include "circ_spec.vh"
module canonical_form(
    input [NUM_CONSTS-1:0] const_inputs,
    input [NUM_VARS-1:0] var_inputs,
    output logic [NUM_OUTPUTS-1:0] outputs
);

    logic [2**NUM_CONSTS-1:0] ohot_const;
    logic [2**NUM_CONSTS-1:0] therm_const;
    logic [2**NUM_VARS-1:0] ohot_vars;

    bin_to_onehot bin_to_ohot_const(.N(2**NUM_CONSTS))(
        .bin(const_inputs),
        .ohot(ohot_const)
    );

    bin_to_onehot bin_to_ohot_var(.N(2**NUM_VARS))(
        .bin(var_inputs),
        .ohot(ohot_vars)
    );

    onehot_to_therm ohot_to_therm_const(.N(2**NUM_CONSTS))(
        .oh(ohot_const),
        .therm(therm_const)
    );

endmodule