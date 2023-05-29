module gb2_opt2(
    input [17:0] x,
    output logic [1:0] z
);
always_comb begin 
	 z = 4'b0000;
	if(x==?18'b1?????????????0000) z |= 2'b10; 
	if(x==?18'b??1???????????1000) z |= 2'b10; 
	if(x==?18'b?1?????????????100) z |= 2'b10; 
	if(x==?18'b???????1??????0010) z |= 2'b10; 
	if(x==?18'b?????????1????1010) z |= 2'b10; 
	if(x==?18'b????????1??????110) z |= 2'b10; 
	if(x==?18'b???1??????????00?1) z |= 2'b10; 
	if(x==?18'b?????1????????10?1) z |= 2'b10; 
	if(x==?18'b????1??????????1?1) z |= 2'b10; 
	if(x==?18'b????1?????????0000) z |= 2'b01; 
	if(x==?18'b??????1???????1000) z |= 2'b01; 
	if(x==?18'b?????1?????????100) z |= 2'b01; 
	if(x==?18'b???????????1??0010) z |= 2'b01; 
	if(x==?18'b?????????????11010) z |= 2'b01; 
	if(x==?18'b????????????1??110) z |= 2'b01; 
	if(x==?18'b????????1?????00?1) z |= 2'b01; 
	if(x==?18'b??????????1???10?1) z |= 2'b01; 
	if(x==?18'b?????????1?????1?1) z |= 2'b01; 
end 
endmodule