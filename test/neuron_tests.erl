-module(neuron_tests).

-include_lib("eunit/include/eunit.hrl").

perceive_test_() ->
    [   fun () ->
            neuron:perceive([neuron:perceive([1,1], [-10, 20, 20]), neuron:perceive([1,1], [-30, 20, 20])], [10, 20, -20]) > 0.9999
        end,
        fun () ->
            neuron:perceive([0,0], [-30,20,20]) < 0.0001
        end,
        fun () ->
            neuron:perceive([0,1], [-30,20,20]) < 0.0001
        end,
        fun () ->
            neuron:perceive([1,0], [-30,20,20]) < 0.0001
        end,
        fun () ->
            neuron:perceive([1,1], [-30,20,20]) > 0.9999
        end
    ].

neuron_gen_server_test_() ->
  {ok, Pid} = neuron:start_link([[0.01, 0.5, 0.2], [{1,0.6}, {2,0.9}], []]),
  L = [fun () ->
    ok = neuron:stimulate(Pid, {1, 0.3}),
    ok = neuron:stimulate(Pid, {1, 0.4}),
    ok = neuron:stimulate(Pid, {2, 0.5})
  end],
  {ok, N1_pid} = neuron:start_link([[0],[],[]]),
  {ok, N2_pid} = neuron:start_link([[0],[],[]]),
  {ok, N3_pid} = neuron:start_link([[0],[],[]]),
  neuron:connect(N1_pid, N2_pid),
  neuron:connect(N1_pid, N3_pid),

  L ++ [
    fun () ->
      neuron:pass(N1_pid, 0.5)
    end
  ].

xor_test_() ->
  {ok, Input1} = neuron:start_link([[0],[],[]]),
  {ok, Input2} = neuron:start_link([[0],[],[]]),

  {ok, N1} = neuron:start_link([[],[],[]]),
  {ok, N2} = neuron:start_link([[],[],[]]),
  {ok, N3} = neuron:start_link([[],[],[]]),

  neuron:connect(Input1, N1),
  neuron:connect(Input2, N1),
  neuron:connect(Input1, N2),
  neuron:connect(Input2, N2),
  neuron:connect(N1, N3),
  neuron:connect(N2, N3),

  neuron:set_weights(N1, [-10, 20, 20]),
  neuron:set_weights(N2, [-30, 20, 20]),
  neuron:set_weights(N3, [10, 20, -20]),

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