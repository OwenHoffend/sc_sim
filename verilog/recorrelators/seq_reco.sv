module seq_reco( //simple 2-state sequential recorrelator
    input x, y, clk, rst_n,
    output logic x_reco_r, y_reco_r
);

localparam INIT = 2'b00,
           SAVE_X = 2'b01,
           SAVE_Y = 2'b10;

logic [1:0] state, next_state;
logic x_reco, y_reco;

//4-ff FSM coding style from http://www.sunburst-design.com/papers/CummingsSNUG2019SV_FSM1.pdf
always_ff @(posedge clk or negedge rst_n)
    if(!rst_n) state <= INIT;
    else       state <= next_state;

always_comb begin : next_state_logic
    next_state = state;
    unique case(state) //unique ensures exactly one case is matched
        INIT: if(x != y) begin
            if(x) next_state = SAVE_X;
            else  next_state = SAVE_Y;
        end
        SAVE_X: if(y && !x) next_state = INIT;
        SAVE_Y: if(x && !y) next_state = INIT;
    endcase
end : next_state_logic

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        x_reco_r <= 1'b0;
        y_reco_r <= 1'b0;
    end else begin
        x_reco_r <= x_reco;
        y_reco_r <= y_reco;
    end
end

always_comb begin : output_logic
    x_reco = x;
    y_reco = y;
    if(x != y) begin
        unique case(state)
            INIT: begin
                x_reco = 1'b0;
                y_reco = 1'b0;
            end
            SAVE_X: if(y) x_reco = 1'b1;
            SAVE_Y: if(x) y_reco = 1'b1;
        endcase
    end
end : output_logic

endmodule