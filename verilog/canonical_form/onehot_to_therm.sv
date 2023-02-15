
//Efficiently convert a one-hot encoding to a thermometer encoding
//Example: 00001000 --> 00001111
//This uses log2(N) layers of logic, recursively defined
module onehot_to_therm #(parameter N = 8, DIR = 0)(
    input [N-1:0] oh,
    output logic [N-1:0] therm
);
    logic [N-1:0] therm_internal;
    generate
    if(N == 2) begin : base
        always_comb begin
            if(DIR) begin //00001000 --> 11111000
                therm[0] = oh[0];
                therm[1] = |oh;
            end else begin //00001000 --> 00001111
                therm[0] = |oh;
                therm[1] = oh[1];
            end
        end
    end else begin : rec
        onehot_to_therm #(.N(N/2), .DIR(DIR)) top(
            .oh(oh[N-1:N/2]),
            .therm(therm_internal[N-1:N/2])
        );
        onehot_to_therm #(.N(N/2), .DIR(DIR)) bot(
            .oh(oh[(N/2)-1:0]),
            .therm(therm_internal[(N/2)-1:0])
        );
        //This looks complicated but all it does is OR half of therm_internal with the first bit of the other half
        always_comb begin 
            if(DIR)
                therm = {{(N/2){therm_internal[(N/2)-1]}} | therm_internal[N-1:N/2], therm_internal[(N/2)-1:0]};
            else
                therm = {therm_internal[N-1:N/2], {(N/2){therm_internal[N/2]}} | therm_internal[(N/2)-1:0]};
        end
    end
    endgenerate
endmodule

`endif