`timescale 1ns/100ps
module seq_reco_d_tb;
logic x, y, clk, rst_n, x_reco_r, y_reco_r;
seq_reco_d #(.DEPTH(1)) DUT(
    .x(x),
    .y(y),
    .clk(clk),
    .rst_n(rst_n),
    .x_reco_r(x_reco_r),
    .y_reco_r(y_reco_r)
);

task fail(
    string signal,
    bit actual_result,
    bit correct_result
);
    $display("TESTCASE FAILED @ time %4.0f: %s caused failure. Was: %b, Should be: %b", $time, signal, actual_result, correct_result);
    $finish;
endtask

task reset();
    rst_n = 1; #1;
    rst_n = 0; #1;
    rst_n = 1; #1;
endtask

task test_bs(
    input logic [255:0] x_test, y_test, x_r_test, y_r_test
);
    reset();
    for(int i = 255; i >=0; i--) begin
        //$display("%d", i);
        x = x_test[i];
        y = y_test[i];
        //$display("%d%d", x, y);
        @(posedge clk);
        @(negedge clk);
        //$display("%d", DUT.state);
        //$display("%d", DUT.sign);
        //$display("%d%d", x_reco_r, y_reco_r);
        //$display("%d%d", x_r_test[i], y_r_test[i]);
        if(x_reco_r != x_r_test[i])
            fail("x", x_reco_r, x_r_test[i]);
        if(y_reco_r != y_r_test[i])
            fail("y", y_reco_r, y_r_test[i]);
    end
    $display("PASSED!");
endtask

always #5 clk = ~clk;

initial begin
    clk = 0; 
    test_bs(x1, y1, x1_r, y1_r);
    test_bs(x2, y2, x2_r, y2_r);
    test_bs(x3, y3, x3_r, y3_r);
    test_bs(x4, y4, x4_r, y4_r);
    $finish;
end
endmodule