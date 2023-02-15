module bin_to_onehot #(parameter N = 8)(
    input [N-1:0] bin,
    output logic [2**N-1:0] ohot
);
    always_comb begin
        ohot = 0;
        ohot[bin] = 1'b1; //might need a recursive form for this, not sure this works on its own
    end
endmodule