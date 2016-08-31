-module(neuron_tests).

-include_lib("eunit/include/eunit.hrl").

perceive_test_() ->
    [   fun () ->
            neuron:perceive([1, neuron:perceive([1,1,1], [-10, 20, 20]), neuron:perceive([1,1,1], [-30, 20, 20])], [10, 20, -20]) > 0.9999
        end,
        fun () ->
            neuron:perceive([1,0,0], [-30,20,20]) < 0.0001
        end,
        fun () ->
            neuron:perceive([1,0,1], [-30,20,20]) < 0.0001
        end,
        fun () ->
            neuron:perceive([1,1,0], [-30,20,20]) < 0.0001
        end,
        fun () ->
            neuron:perceive([1,1,1], [-30,20,20]) > 0.9999
        end
    ].

neuron_gen_server_test_() ->
  {ok, N1} = neuron:start_link(),
  {ok, N2} = neuron:start_link(),
  {ok, N3} = neuron:start_link(),
  {ok, N4} = neuron:start_link(),

  neuron:connect(N1, N2),
  neuron:connect(N1, N3),
  neuron:connect(N2, N4),
  neuron:connect(N3, N4),
  [
    fun () ->
      neuron:pass(N1, 0.5)
    end
  ].

xor_test_() ->
  {ok, Input1} = neuron:start_link(),
  {ok, Input2} = neuron:start_link(),
  {ok, N1} = neuron:start_link(),
  {ok, N2} = neuron:start_link(),
  {ok, N3} = neuron:start_link(),

  neuron:connect(Input1, N1),
  neuron:connect(Input2, N1),
  neuron:connect(Input1, N2),
  neuron:connect(Input2, N2),
  neuron:connect(N1, N3),
  neuron:connect(N2, N3),

  neuron:set_weights(N1, [{1, -30}, {Input1, 20}, {Input2, 20}]),
  neuron:set_weights(N2, [{1,  10}, {Input1,-20}, {Input2,-20}]),
  neuron:set_weights(N3, [{1, -10}, {N1,     20}, {N2,     20}]),

  [
    fun () ->
      neuron:pass(Input1, 0),
      neuron:pass(Input2, 1),
      timer:sleep(100),
      ?assert(neuron:get_output(N3) < 0.0001)
    end,
    fun () ->
      neuron:pass(Input1, 0),
      neuron:pass(Input2, 0),
      timer:sleep(100),
      ?assert(neuron:get_output(N3) > 0.999)
    end
  ].