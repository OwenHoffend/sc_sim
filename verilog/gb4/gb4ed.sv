
//Main configuration
//`define RECO 7
`define OPT

module gb4ed(
    input [20:0] x,
    input clk, rst_n,
    output logic z
);

    logic [3:0] gb4_out;
`ifdef OPT
    //gb2_opt gb2_top(
    //    .x({
    //        x[20], x[19], x[18],
    //        x[16], x[15], x[14], x[13],
    //        x[12], x[11], x[10], x[9],
    //               x[7],  x[6],  x[5],
    //        x[4],  x[3],  x[2],  x[1]
    //    }),
    //    .z({gb4_out[0], gb4_out[3]})
    //);
    //gb2_opt gb2_bot(
    //    .x({
    //        x[17], x[18], x[19],
    //        x[13], x[14], x[15], x[16],
    //        x[9],  x[10], x[11], x[12],
    //               x[6],  x[7],  x[8],
    //        x[4],  x[3],  x[2],  x[1]
    //    }),
    //    .z({gb4_out[1], gb4_out[2]})
    //);
    gb4_opt gb41_opt(
        .x(x[20:1]),
        .z(gb4_out)
    );
`else
    gb4 gb41(
        .x(x[20:1]),
        .z(gb4_out)
    );
`endif

`ifdef RECO
    logic [3:0] gb4_out_reco;
    seq_reco_d #(.DEPTH(`RECO)) reco_top (
        .x(gb4_out[0]), 
        .y(gb4_out[3]), 
        .clk(clk), 
        .rst_n(rst_n),
        .x_reco_r(gb4_out_reco[0]), 
        .y_reco_r(gb4_out_reco[3])
    );
    seq_reco_d #(.DEPTH(`RECO)) reco_bot (
        .x(gb4_out[1]), 
        .y(gb4_out[2]), 
        .clk(clk), 
        .rst_n(rst_n),
        .x_reco_r(gb4_out_reco[1]), 
        .y_reco_r(gb4_out_reco[2])
    );
`endif

    rced rced1(
`ifdef RECO
        .x(gb4_out_reco),
`else
        .x(gb4_out),
`endif
        .c(x[0]),
        .z(z)
    );
endmodule