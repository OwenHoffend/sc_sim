`ifndef CANONICAL_FORM_TB
`define CANONICAL_FORM_TB
`include "./sc_sim/verilog/canonical_form/circs/gb4.svh"
module canonical_form_tb;
  
  // Declare signals
  logic [NUM_CONSTS-1:0] const_inputs;
  logic [NUM_VARS-1:0] var_inputs;
  logic [NUM_OUTPUTS-1:0] outputs;
  logic clock;
  
  // Instantiate the module under test
  canonical_form dut(
    .const_inputs(const_inputs),
    .var_inputs(var_inputs),
    .outputs(outputs)
  );

  always begin
      #5 clock = ~clock;
  end

  integer weights [NUM_OUTPUTS-1:0];
  initial begin
    $monitor("Time:%4.0f, var_inputs:%b, const_inputs:%b, outputs:%b, ohot_const:%b", 
      $time, var_inputs, const_inputs, outputs, dut.ohot_const);
    clock = 0;
    var_inputs = 0;
    const_inputs = 0;
    for(int v = 0; v < 2**NUM_VARS; v++) begin
      for(int k = 0; k < NUM_OUTPUTS; k++)begin
        weights[k] = 0;
      end
      for(int i = 0; i < 2**NUM_CONSTS; i++) begin
        @(negedge clock); //Design is combinational, but this helps divide tests for debugging
        const_inputs++;
        for(int k = 0; k < NUM_OUTPUTS; k++)begin
          if(outputs[k])
            weights[k]++;
        end
      end
      for(int k = 0; k < NUM_OUTPUTS; k++) begin
        $write("%d", weights[k]);
        assert(weights[k] == WEIGHT_MATRIX[k*(2**NUM_VARS)+v]);
      end
      $write("\n");
      var_inputs++;
    end
    $finish;

  end

endmodule
`endif