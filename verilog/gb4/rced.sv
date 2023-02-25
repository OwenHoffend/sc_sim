module rced(
    input x[3:0], 
    input c,
    output logic z
);
    logic mux1, mux2;
    assign mux1 = x[0] ^ x[3];
    assign mux2 = x[1] ^ x[2];
    assign z = c ? mux1 : mux2;
endmodule