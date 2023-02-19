//Sequential recorrelator with parameterizable depth
module seq_reco_d #(parameter DEPTH=1)(
    input x, y, clk, rst_n,
    output logic x_reco_r, y_reco_r
);

logic [$clog2(DEPTH+1)-1:0] state, next_state;
logic sign, next_sign;
logic x_reco, y_reco;

always_ff @(posedge clk or negedge rst_n) begin //next state registers
    if(!rst_n) begin
        state <= INIT;
        sign <= 1'b0;
    end else begin 
        state <= next_state;
        sign <= next_sign;
    end
end

always_ff @(posedge clk or negedge rst_n) begin //output registers
    if(!rst_n) begin
        x_reco_r <= 1'b0;
        y_reco_r <= 1'b0;
    end else begin
        x_reco_r <= x_reco;
        y_reco_r <= y_reco;
    end
end

always_comb begin //next state logic
    next_state = state;
    next_sign = sign;
    if(x != y) begin
        if(state == 0) begin
            next_state = state + 1'b1;
            if(x) next_sign = 1'b1;
            else next_sign = 1'b0;
        end else begin
            if(sign) begin //left side
                if(y) next_state = state - 1'b1;
                else if(state < DEPTH) next_state = state + 1'b1; 
            end else begin //right side
                if(x) next_state = state - 1'b1;
                else if(state < DEPTH) next_state = state + 1'b1; 
            end 
        end
    end
end

always_comb begin //output logic
    x_reco = x;
    y_reco = y;
    if(x != y) begin
        if(sign) begin //left side
            if(y) x_reco = 1'b1;
            else if(state < DEPTH) x_reco = 1'b0;
        end else begin //right side
            if(x) y_reco = 1'b1;
            else if(state < DEPTH) y_reco = 1'b0;
        end 
    end
end
endmodule