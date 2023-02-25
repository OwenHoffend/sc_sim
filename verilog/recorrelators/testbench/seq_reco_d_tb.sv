`timescale 1ns/10ps
`include "./sc_sim/verilog/recorrelators/testbench/reco_test_cases.svh"
module seq_reco_d_tb;

logic clk, rst_n, x_r_test, y_r_test;
`ifdef SYNTHESIS
logic x, y, x_reco_r, y_reco_r;
    seq_reco_d DUT (
        .x(x),
        .y(y),
        .clk(clk),
        .rst_n(rst_n),
        .x_reco_r(x_reco_r),
        .y_reco_r(y_reco_r)
    );
`else
logic [NUM_DEPTHS-1:0] x, y, x_reco_r, y_reco_r;
genvar d;
generate
    for(d = 0; d < NUM_DEPTHS; d++) begin : seq_reco_d_instance
        seq_reco_d #(.DEPTH(d+1)) DUT (
            .x(x[d]),
            .y(y[d]),
            .clk(clk),
            .rst_n(rst_n),
            .x_reco_r(x_reco_r[d]),
            .y_reco_r(y_reco_r[d])
        );
    end
endgenerate
`endif

//Create the clock
always #10 clk = ~clk;
initial begin
    clk = 0;
end

task fail(
    string signal,
    bit actual_result,
    bit correct_result
);
    $display("TESTCASE FAILED @ time %4.0f: %s caused failure. Was: %b, Should be: %b", $time, signal, actual_result, correct_result);
    $finish;
endtask

initial begin
    rst_n = 1; #1
    rst_n = 0; #1
    rst_n = 1; #1
`ifdef SYNTHESIS
    for(int i = N-1; i >=0; i--) begin
        //$display("%d", i);
        x        = tcs[(`DEFAULT_DEPTH-1)*4][i];
        y        = tcs[(`DEFAULT_DEPTH-1)*4+1][i];
        x_r_test = tcs[(`DEFAULT_DEPTH-1)*4+2][i];
        y_r_test = tcs[(`DEFAULT_DEPTH-1)*4+3][i];
        //$display("%d%d", x, y);
        @(posedge clk);
        @(negedge clk);
        //$display("%d%d", x_reco_r, y_reco_r);
        //$display("%d%d", x_r_test, y_r_test);
        if(x_reco_r !== x_r_test)
            fail("x", x_reco_r, x_r_test);
        if(y_reco_r !== y_r_test)
            fail("y", y_reco_r, y_r_test);
    end
`else
    for(int d = 0; d < NUM_DEPTHS; d++) begin
        $display("Depth: %d", d);
        for(int i = N-1; i >=0; i--) begin
            //$display("%d", i);
            x[d]     = tcs[4*d][i];
            y[d]     = tcs[4*d+1][i];
            x_r_test = tcs[4*d+2][i];
            y_r_test = tcs[4*d+3][i];
            //$display("%d%d", x[d], y[d]);
            @(posedge clk);
            @(negedge clk);
            //$display("%d%d", x_reco_r[d], y_reco_r[d]);
            //$display("%d%d", x_r_test, y_r_test);
            if(x_reco_r[d] !== x_r_test)
                fail("x", x_reco_r[d], x_r_test);
            if(y_reco_r[d] !== y_r_test)
                fail("y", y_reco_r[d], y_r_test);
        end
    end
`endif
    $display("PASSED!");
    $finish;
end
endmodule