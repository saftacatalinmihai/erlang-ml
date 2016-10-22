-module(neuron_loop_tests).

-include_lib("eunit/include/eunit.hrl").

send_all_test() ->
  {ok, I1} = neuron_loop:start(),
  {ok, I2} = neuron_loop:start(),
  {ok, I3} = neuron_loop:start(),
  {ok, I4} = neuron_loop:start(),
  {ok, I5} = neuron_loop:start(),

  io:format("~p~n", [[I1, I2, I3, I4, I5]]),

  Resp = neuron_loop:send_all([I1, I2, I3, I4, I5], ping),
  io:format("~p~n", [Resp]),
  ?assertEqual([{ok,pong}, {ok, pong}, {ok,pong}, {ok, pong}, {ok,pong}], Resp).


nxor_test() ->
  io:format("Start Test~n"),

  {ok, I1} = neuron_loop:start(),
  {ok, I2} = neuron_loop:start(),
  {ok, H1} = neuron_loop:start(),
  {ok, H2} = neuron_loop:start(),
  {ok, O} = neuron_loop:start(),

  neuron_loop:addOutputs(I1, [H1, H2]),
  neuron_loop:addOutputs(I2, [H1, H2]),
  neuron_loop:addOutputs(H1, [O]),
  neuron_loop:addOutputs(H2, [O]),

  neuron_loop:setBias(H1, -30),
  neuron_loop:setWeight(H1,  I1, 20),
  neuron_loop:setWeight(H1,  I2, 20),


  neuron_loop:setBias(H2, 10),
  neuron_loop:setWeight(H2,  I1, -20),
  neuron_loop:setWeight(H2,  I2, -20),

  neuron_loop:setBias(O, -10),
  neuron_loop:setWeight(O,  H1, 20),
  neuron_loop:setWeight(O,  H2, 20),

  neuron_loop:pass(I1, 0),
  neuron_loop:pass(I2, 0),
  timer:sleep(1000),
  A1 = neuron_loop:getActivation(O),
  ?debugFmt("0 - 0 -> 1 ~p~n", [A1]),
  ?assert(A1 > 0.9),

  neuron_loop:pass(I1, 1),
  neuron_loop:pass(I2, 1),
  timer:sleep(1000),
  A2 = neuron_loop:getActivation(O),
  ?debugFmt("1 - 1 -> 1 ~p~n", [A2]),
  ?assert(A2 > 0.9),

  neuron_loop:pass(I1, 0),
  neuron_loop:pass(I2, 1),
  timer:sleep(1000),
  A3 = neuron_loop:getActivation(O),
  ?debugFmt("0 - 1 -> 0: ~p~n", [A3]),
  ?assert(A3 < 0.1),

  neuron_loop:pass(I1, 1),
  neuron_loop:pass(I2, 0),
  timer:sleep(1000),
  A4 = neuron_loop:getActivation(O),
  ?debugFmt("1 - 0 -> 0: ~p~n", [A4]),
  ?assert(A4 < 0.1).
