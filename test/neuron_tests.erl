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
      neuron:pass(Input1, 0, call),
      neuron:pass(Input2, 1, call),
      ?assert(neuron:get_output(N3) < 0.0001)
    end,
    fun () ->
      neuron:pass(Input1, 0, call),
      neuron:pass(Input2, 0, call),
      ?assert(neuron:get_output(N3) > 0.999)
    end
  ].

learn_single_neuron_test_() ->
  {ok, Input1} = neuron:start_link(),
  {ok, Input2} = neuron:start_link(),
  {ok, N1} = neuron:start_link(),

  neuron:connect(Input1, N1),
  neuron:connect(Input2, N1),

  F = fun (I1,I2, E) ->
    neuron:pass(Input1, I1, call),
    neuron:pass(Input2, I2, call),
    neuron:learn(N1, E)
  end,

  L = [
    fun() -> F(1,1,1) end,
    fun() -> F(1,0,0) end,
    fun() -> F(0,1,0) end,
    fun() -> F(0,0,0) end
  ],

  L2 = lists:flatten([ L ++ L || _ <- lists:seq(1, 1000)]),
  L3 = [X||{_,X} <- lists:sort([ {rand:uniform(), N} || N <- L2])],

  lists:foreach(
    fun(Fn) -> Fn() end,
    L3
  ),

  [
    fun () ->
      neuron:pass(Input1, 0, call),
      neuron:pass(Input2, 1, call),
      ?assert(neuron:get_output(N1) < 0.1)
    end,
    fun () ->
      neuron:pass(Input1, 1, call),
      neuron:pass(Input2, 1, call),
      ?assert(neuron:get_output(N1) > 0.9)
    end
  ].

learn_nxor_test_() ->
  <<A:32, B:32, C:32>> = crypto:rand_bytes(12),
  random:seed({A,B,C}),

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

%%  timer:sleep(100),

  F = fun (I1,I2, E) ->
    neuron:pass(Input1, I1, call),
    neuron:pass(Input2, I2, call),
    neuron:learn(N3, E)
  end,

  L = [
    fun() -> F(1,1,1) end,
    fun() -> F(1,0,0) end,
    fun() -> F(0,1,0) end,
    fun() -> F(0,0,1) end
  ],

  L2 = lists:flatten([ L ++ L || _ <- lists:seq(1, 1000)]),
  L3 = [X||{_,X} <- lists:sort([ {rand:uniform(), N} || N <- L2])],

  lists:foreach(
    fun(Fn) -> Fn() end,
    L3
  ),
%%  timer:sleep(100),
%%  ?debugFmt("N1: ~p~n", [neuron:get_state(N1)]),
%%  ?debugFmt("N2: ~p~n", [neuron:get_state(N2)]),
%%  ?debugFmt("N3: ~p~n", [neuron:get_state(N3)]),

  [
    fun () ->
      ok = neuron:pass(Input1, 0, call),
      ok = neuron:pass(Input2, 1, call),
      O = neuron:get_output(N3),
%%      ?debugFmt("O 0: ~p~n", [O]),
      ?assert( O < 0.1)
    end,
    fun () ->
      ok = neuron:pass(Input1, 0, call),
      ok = neuron:pass(Input2, 0, call),
      O = neuron:get_output(N3),
%%      ?debugFmt("O 1: ~p~n", [O]),
      ?assert( O > 0.9)

    end
  ].

learn_7seg_digits_test_() ->
  <<A:32, B:32, C:32>> = crypto:rand_bytes(12),
  random:seed({A,B,C}),

  Inputs  = neuron:start_multi(7),
  Hidden1  = neuron:start_multi(7),
%%  Hidden2  = neuron:start_multi(10),
  Outputs = neuron:start_multi(10),

  neuron:full_connect(Inputs, Hidden1),
%%  neuron:full_connect(Hidden1, Hidden2),
  neuron:full_connect(Hidden1, Outputs),

  Learning_example = fun
     (0) ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 0, call),
      neuron:pass(lists:nth(5, Inputs), 1, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      neuron:learn(lists:nth(1, Outputs), 0),
      neuron:learn(lists:nth(2, Outputs), 0),
      neuron:learn(lists:nth(3, Outputs), 0),
      neuron:learn(lists:nth(4, Outputs), 0),
      neuron:learn(lists:nth(5, Outputs), 0),
      neuron:learn(lists:nth(6, Outputs), 0),
      neuron:learn(lists:nth(7, Outputs), 0),
      neuron:learn(lists:nth(8, Outputs), 0),
      neuron:learn(lists:nth(9, Outputs), 0),
      neuron:learn(lists:nth(10, Outputs),1);
    (1) ->
      neuron:pass(lists:nth(1, Inputs), 0, call),
      neuron:pass(lists:nth(2, Inputs), 0, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 0, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 0, call),

      neuron:learn(lists:nth(1, Outputs), 1),
      neuron:learn(lists:nth(2, Outputs), 0),
      neuron:learn(lists:nth(3, Outputs), 0),
      neuron:learn(lists:nth(4, Outputs), 0),
      neuron:learn(lists:nth(5, Outputs), 0),
      neuron:learn(lists:nth(6, Outputs), 0),
      neuron:learn(lists:nth(7, Outputs), 0),
      neuron:learn(lists:nth(8, Outputs), 0),
      neuron:learn(lists:nth(9, Outputs), 0),
      neuron:learn(lists:nth(10, Outputs),0);
    (2) ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 0, call),
      neuron:pass(lists:nth(3, Inputs), 0, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 1, call),
      neuron:pass(lists:nth(6, Inputs), 0, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      neuron:learn(lists:nth(1, Outputs), 0),
      neuron:learn(lists:nth(2, Outputs), 1),
      neuron:learn(lists:nth(3, Outputs), 0),
      neuron:learn(lists:nth(4, Outputs), 0),
      neuron:learn(lists:nth(5, Outputs), 0),
      neuron:learn(lists:nth(6, Outputs), 0),
      neuron:learn(lists:nth(7, Outputs), 0),
      neuron:learn(lists:nth(8, Outputs), 0),
      neuron:learn(lists:nth(9, Outputs), 0),
      neuron:learn(lists:nth(10, Outputs),0);
    (3) ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 0, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      neuron:learn(lists:nth(1, Outputs), 0),
      neuron:learn(lists:nth(2, Outputs), 0),
      neuron:learn(lists:nth(3, Outputs), 1),
      neuron:learn(lists:nth(4, Outputs), 0),
      neuron:learn(lists:nth(5, Outputs), 0),
      neuron:learn(lists:nth(6, Outputs), 0),
      neuron:learn(lists:nth(7, Outputs), 0),
      neuron:learn(lists:nth(8, Outputs), 0),
      neuron:learn(lists:nth(9, Outputs), 0),
      neuron:learn(lists:nth(10, Outputs),0);
    (4) ->
      neuron:pass(lists:nth(1, Inputs), 0, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 0, call),

      neuron:learn(lists:nth(1, Outputs), 0),
      neuron:learn(lists:nth(2, Outputs), 0),
      neuron:learn(lists:nth(3, Outputs), 0),
      neuron:learn(lists:nth(4, Outputs), 1),
      neuron:learn(lists:nth(5, Outputs), 0),
      neuron:learn(lists:nth(6, Outputs), 0),
      neuron:learn(lists:nth(7, Outputs), 0),
      neuron:learn(lists:nth(8, Outputs), 0),
      neuron:learn(lists:nth(9, Outputs), 0),
      neuron:learn(lists:nth(10, Outputs),0);
    (5) ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 0, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      neuron:learn(lists:nth(1, Outputs), 0),
      neuron:learn(lists:nth(2, Outputs), 0),
      neuron:learn(lists:nth(3, Outputs), 0),
      neuron:learn(lists:nth(4, Outputs), 0),
      neuron:learn(lists:nth(5, Outputs), 1),
      neuron:learn(lists:nth(6, Outputs), 0),
      neuron:learn(lists:nth(7, Outputs), 0),
      neuron:learn(lists:nth(8, Outputs), 0),
      neuron:learn(lists:nth(9, Outputs), 0),
      neuron:learn(lists:nth(10, Outputs),0);
    (6) ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 0, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 1, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      neuron:learn(lists:nth(1, Outputs), 0),
      neuron:learn(lists:nth(2, Outputs), 0),
      neuron:learn(lists:nth(3, Outputs), 0),
      neuron:learn(lists:nth(4, Outputs), 0),
      neuron:learn(lists:nth(5, Outputs), 0),
      neuron:learn(lists:nth(6, Outputs), 1),
      neuron:learn(lists:nth(7, Outputs), 0),
      neuron:learn(lists:nth(8, Outputs), 0),
      neuron:learn(lists:nth(9, Outputs), 0),
      neuron:learn(lists:nth(10, Outputs),0);
    (7) ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 0, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 0, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 0, call),

      neuron:learn(lists:nth(1, Outputs), 0),
      neuron:learn(lists:nth(2, Outputs), 0),
      neuron:learn(lists:nth(3, Outputs), 0),
      neuron:learn(lists:nth(4, Outputs), 0),
      neuron:learn(lists:nth(5, Outputs), 0),
      neuron:learn(lists:nth(6, Outputs), 0),
      neuron:learn(lists:nth(7, Outputs), 1),
      neuron:learn(lists:nth(8, Outputs), 0),
      neuron:learn(lists:nth(9, Outputs), 0),
      neuron:learn(lists:nth(10, Outputs),0);
    (8) ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 1, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      neuron:learn(lists:nth(1, Outputs), 0),
      neuron:learn(lists:nth(2, Outputs), 0),
      neuron:learn(lists:nth(3, Outputs), 0),
      neuron:learn(lists:nth(4, Outputs), 0),
      neuron:learn(lists:nth(5, Outputs), 0),
      neuron:learn(lists:nth(6, Outputs), 0),
      neuron:learn(lists:nth(7, Outputs), 0),
      neuron:learn(lists:nth(8, Outputs), 1),
      neuron:learn(lists:nth(9, Outputs), 0),
      neuron:learn(lists:nth(10, Outputs),0);
    (9) ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 0, call),

      neuron:learn(lists:nth(1, Outputs), 0),
      neuron:learn(lists:nth(2, Outputs), 0),
      neuron:learn(lists:nth(3, Outputs), 0),
      neuron:learn(lists:nth(4, Outputs), 0),
      neuron:learn(lists:nth(5, Outputs), 0),
      neuron:learn(lists:nth(6, Outputs), 0),
      neuron:learn(lists:nth(7, Outputs), 0),
      neuron:learn(lists:nth(8, Outputs), 0),
      neuron:learn(lists:nth(9, Outputs), 1),
      neuron:learn(lists:nth(10, Outputs),0)
  end,

  L = [
    fun() -> Learning_example(1) end,
    fun() -> Learning_example(2) end,
    fun() -> Learning_example(3) end,
    fun() -> Learning_example(4) end,
    fun() -> Learning_example(5) end,
    fun() -> Learning_example(6) end,
    fun() -> Learning_example(7) end,
    fun() -> Learning_example(8) end,
    fun() -> Learning_example(9) end,
    fun() -> Learning_example(0) end

  ],

  L2 = lists:flatten([ L ++ L || _ <- lists:seq(1, 1000)]),
  L3 = [X||{_,X} <- lists:sort([ {rand:uniform(), N} || N <- L2])],

  lists:foreach(
    fun(Fn) -> Fn() end,
    L3
  ),

  [
    fun () ->
      neuron:pass(lists:nth(1, Inputs), 0, call),
      neuron:pass(lists:nth(2, Inputs), 0, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 0, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 0, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) > 0.9 ),
      ?assert(lists:nth(2, Results) < 0.1 ),
      ?assert(lists:nth(3, Results) < 0.1 ),
      ?assert(lists:nth(4, Results) < 0.1 ),
      ?assert(lists:nth(5, Results) < 0.1 ),
      ?assert(lists:nth(6, Results) < 0.1 ),
      ?assert(lists:nth(7, Results) < 0.1 ),
      ?assert(lists:nth(8, Results) < 0.1 ),
      ?assert(lists:nth(9, Results) < 0.1 ),
      ?assert(lists:nth(10, Results) < 0.1)
    end,
    fun () ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 0, call),
      neuron:pass(lists:nth(3, Inputs), 0, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 1, call),
      neuron:pass(lists:nth(6, Inputs), 0, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) < 0.1 ),
      ?assert(lists:nth(2, Results) > 0.9 ),
      ?assert(lists:nth(3, Results) < 0.1 ),
      ?assert(lists:nth(4, Results) < 0.1 ),
      ?assert(lists:nth(5, Results) < 0.1 ),
      ?assert(lists:nth(6, Results) < 0.1 ),
      ?assert(lists:nth(7, Results) < 0.1 ),
      ?assert(lists:nth(8, Results) < 0.1 ),
      ?assert(lists:nth(9, Results) < 0.1 ),
      ?assert(lists:nth(10, Results) < 0.1)
    end,
    fun() ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 0, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) < 0.1 ),
      ?assert(lists:nth(2, Results) < 0.1 ),
      ?assert(lists:nth(3, Results) > 0.9 ),
      ?assert(lists:nth(4, Results) < 0.1 ),
      ?assert(lists:nth(5, Results) < 0.1 ),
      ?assert(lists:nth(6, Results) < 0.1 ),
      ?assert(lists:nth(7, Results) < 0.1 ),
      ?assert(lists:nth(8, Results) < 0.1 ),
      ?assert(lists:nth(9, Results) < 0.1 ),
      ?assert(lists:nth(10, Results) < 0.1)
    end,
    fun() ->
      neuron:pass(lists:nth(1, Inputs), 0, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 0, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) < 0.1 ),
      ?assert(lists:nth(2, Results) < 0.1 ),
      ?assert(lists:nth(3, Results) < 0.1 ),
      ?assert(lists:nth(4, Results) > 0.9 ),
      ?assert(lists:nth(5, Results) < 0.1 ),
      ?assert(lists:nth(6, Results) < 0.1 ),
      ?assert(lists:nth(7, Results) < 0.1 ),
      ?assert(lists:nth(8, Results) < 0.1 ),
      ?assert(lists:nth(9, Results) < 0.1 ),
      ?assert(lists:nth(10, Results) < 0.1)
    end,
    fun() ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 0, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) < 0.1 ),
      ?assert(lists:nth(2, Results) < 0.1 ),
      ?assert(lists:nth(3, Results) < 0.1 ),
      ?assert(lists:nth(4, Results) < 0.1 ),
      ?assert(lists:nth(5, Results) > 0.9 ),
      ?assert(lists:nth(6, Results) < 0.1 ),
      ?assert(lists:nth(7, Results) < 0.1 ),
      ?assert(lists:nth(8, Results) < 0.1 ),
      ?assert(lists:nth(9, Results) < 0.1 ),
      ?assert(lists:nth(10, Results) < 0.1)
    end,
    fun() ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 0, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 1, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) < 0.1 ),
      ?assert(lists:nth(2, Results) < 0.1 ),
      ?assert(lists:nth(3, Results) < 0.1 ),
      ?assert(lists:nth(4, Results) < 0.1 ),
      ?assert(lists:nth(5, Results) < 0.1 ),
      ?assert(lists:nth(6, Results) > 0.9 ),
      ?assert(lists:nth(7, Results) < 0.1 ),
      ?assert(lists:nth(8, Results) < 0.1 ),
      ?assert(lists:nth(9, Results) < 0.1 ),
      ?assert(lists:nth(10, Results) < 0.1)
    end,
    fun() ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 0, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 0, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 0, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) < 0.1 ),
      ?assert(lists:nth(2, Results) < 0.1 ),
      ?assert(lists:nth(3, Results) < 0.1 ),
      ?assert(lists:nth(4, Results) < 0.1 ),
      ?assert(lists:nth(5, Results) < 0.1 ),
      ?assert(lists:nth(6, Results) < 0.1 ),
      ?assert(lists:nth(7, Results) > 0.9 ),
      ?assert(lists:nth(8, Results) < 0.1 ),
      ?assert(lists:nth(9, Results) < 0.1 ),
      ?assert(lists:nth(10, Results) < 0.1)
    end,
    fun() ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 1, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) < 0.1 ),
      ?assert(lists:nth(2, Results) < 0.1 ),
      ?assert(lists:nth(3, Results) < 0.1 ),
      ?assert(lists:nth(4, Results) < 0.1 ),
      ?assert(lists:nth(5, Results) < 0.1 ),
      ?assert(lists:nth(6, Results) < 0.1 ),
      ?assert(lists:nth(7, Results) < 0.1 ),
      ?assert(lists:nth(8, Results) > 0.9 ),
      ?assert(lists:nth(9, Results) < 0.1 ),
      ?assert(lists:nth(10, Results) < 0.2)
    end,
    fun() ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 1, call),
      neuron:pass(lists:nth(5, Inputs), 0, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 0, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) < 0.1 ),
      ?assert(lists:nth(2, Results) < 0.1 ),
      ?assert(lists:nth(3, Results) < 0.1 ),
      ?assert(lists:nth(4, Results) < 0.2 ),
      ?assert(lists:nth(5, Results) < 0.1 ),
      ?assert(lists:nth(6, Results) < 0.1 ),
      ?assert(lists:nth(7, Results) < 0.1 ),
      ?assert(lists:nth(8, Results) < 0.1 ),
      ?assert(lists:nth(9, Results) > 0.9 ),
      ?assert(lists:nth(10, Results) < 0.1)
    end,
    fun() ->
      neuron:pass(lists:nth(1, Inputs), 1, call),
      neuron:pass(lists:nth(2, Inputs), 1, call),
      neuron:pass(lists:nth(3, Inputs), 1, call),
      neuron:pass(lists:nth(4, Inputs), 0, call),
      neuron:pass(lists:nth(5, Inputs), 1, call),
      neuron:pass(lists:nth(6, Inputs), 1, call),
      neuron:pass(lists:nth(7, Inputs), 1, call),

      Results = lists:map(fun(O) -> neuron:get_output(O) end, Outputs),

      ?assert(lists:nth(1, Results) < 0.1 ),
      ?assert(lists:nth(2, Results) < 0.1 ),
      ?assert(lists:nth(3, Results) < 0.1 ),
      ?assert(lists:nth(4, Results) < 0.1 ),
      ?assert(lists:nth(5, Results) < 0.1 ),
      ?assert(lists:nth(6, Results) < 0.1 ),
      ?assert(lists:nth(7, Results) < 0.1 ),
      ?assert(lists:nth(8, Results) < 0.2 ),
      ?assert(lists:nth(9, Results) < 0.1 ),
      ?assert(lists:nth(10, Results) > 0.9)
    end
  ].
