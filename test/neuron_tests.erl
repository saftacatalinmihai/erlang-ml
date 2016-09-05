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
  {ok, N3} = neuron:start_link(fun ([Pid, Out]) -> io:format("~p: ~p~n", [Pid, Out]) end),

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
      neuron:pass(Input1, {predict, 0}),
      neuron:pass(Input2, {predict, 1}),
      ?assert(neuron:get_output(N3) < 0.0001)
    end,
    fun () ->
      neuron:pass(Input1, {predict, 0}),
      neuron:pass(Input2, {predict, 0}),
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
  <<A:32, B:32, C:32>> = crypto:strong_rand_bytes(12),
  random:seed({A,B,C}),

  {ok, Input1} = neuron:start_link(),
  {ok, Input2} = neuron:start_link(),
  {ok, N1} = neuron:start_link(),
  {ok, N2} = neuron:start_link(),
  {ok, N3} = neuron:start_link(fun ([Pid, Out]) -> io:format("~p ~p~n", [Pid, Out]) end),

  neuron:connect(Input1, N1),
  neuron:connect(Input2, N1),

  neuron:connect(Input1, N2),
  neuron:connect(Input2, N2),

  neuron:connect(N1, N3),
  neuron:connect(N2, N3),

%%  timer:sleep(100),

  F = fun (I1,I2, Expected) ->
    Example_ref = make_ref(),
    neuron:pass(Input1, {train, Example_ref, I1}),
    neuron:pass(Input2, {train, Example_ref, I2}),
    neuron:learn(N3, {Example_ref, Expected})
  end,

  L = [
    fun() -> F(1,1,1) end,
    fun() -> F(1,0,1) end,
    fun() -> F(0,1,0) end,
    fun() -> F(0,0,0) end
  ],

  L2 = lists:flatten([ L ++ L || _ <- lists:seq(1, 10)]),
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
  <<A:32, B:32, C:32>> = crypto:strong_rand_bytes(12),
  random:seed({A,B,C}),

  Inputs  = neuron:start_multi(7),
  Hidden1  = neuron:start_multi(7),
%%  Hidden2  = neuron:start_multi(10),
  Outputs = neuron:start_multi(10), %%, fun ([Pid, Out]) -> io:format("~p~p~n", [Pid, Out]) end),

  neuron:full_connect(Inputs, Hidden1),
%%  neuron:full_connect(Hidden1, Hidden2),
  neuron:full_connect(Hidden1, Outputs),

  NumToVec = fun
               (0) -> [1,1,1,0,1,1,1];
               (1) -> [0,0,1,0,0,1,0];
               (2) -> [1,0,0,1,1,0,1];
               (3) -> [1,0,1,1,0,1,1];
               (4) -> [0,1,1,1,0,1,0];
               (5) -> [1,1,0,1,0,1,1];
               (6) -> [1,1,0,1,1,1,1];
               (7) -> [1,0,1,0,0,1,0];
               (8) -> [1,1,1,1,1,1,1];
               (9) -> [1,1,1,1,0,1,0]
            end,
  NumToOut = fun
               (0) -> [0,0,0,0,0,0,0,0,0,1];
               (I) -> lists:map(fun(_) -> 0 end,lists:seq(1, I-1)) ++ [1] ++ lists:map(fun(_) -> 0 end,lists:seq(I+1, 10))
             end,

  Learning_example = fun (N) ->
      lists:foreach(
        fun ({Input, InVal}) ->
          neuron:pass(Input, InVal, call)
        end,
        lists:zip(Inputs, NumToVec(N))
      ),

      lists:foreach(
        fun ({Output, OutVal}) ->
          neuron:learn(Output, OutVal)
        end,
        lists:zip(Outputs, NumToOut(N))
      )
  end,

  L = lists:map( fun (I) -> fun () -> Learning_example(I) end end, lists:seq(0,9)),
  L2 = lists:flatten([ L ++ L || _ <- lists:seq(1, 500)]),
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

learn_mnist_digits_() ->
  <<A:32, B:32, C:32>> = crypto:strong_rand_bytes(12),
  random:seed({A,B,C}),

  Inputs  = neuron:start_multi(784),
  Hidden1 = neuron:start_multi(300),
  Outputs = neuron:start_multi(10),

  neuron:full_connect(Inputs, Hidden1),
  neuron:full_connect(Hidden1, Outputs),

  TrainingSet = csv_parser:parse("train_5000.csv"),

  Label_To_Learn = fun(Label) ->
    case Label of
      0 -> neuron:learn(lists:nth(10, Outputs), 1);
      _ -> neuron:learn(lists:nth(10, Outputs), 0)
    end,

    lists:foreach(
      fun(Idx) ->

        neuron:learn(
          lists:nth(Idx, Outputs),
          case Label =:= Idx of
            true -> 1;
            false -> 0
          end
        )
      end,
      lists:seq(1,9)
    )
  end,

  lists:foldl(
    fun([Label | In], Count) ->
      io:format("Learning idx ~p: ~p~n", [Count, Label]),
      TrainingExamples = lists:zip(Inputs, In),
      lists:foreach(fun({Neuron, Inp}) -> neuron:pass(Neuron, Inp) end, tl(TrainingExamples) ),
      {N,I} = hd(TrainingExamples),
      neuron:pass(N,I, call),

      Label_To_Learn(Label),
      Count + 1
    end,
    1,
    TrainingSet
  ),

%%  Test_Set = csv_parser:parse("test_1000.csv"),

  lists:foldl(
    fun([Label | In], Acc) ->
      lists:foreach(fun({Neuron, Inp}) -> neuron:pass(Neuron, Inp, call) end, lists:zip(Inputs, In)),

      Predictions = lists:map(fun({Idx, N}) -> {Idx, neuron:get_output(N)} end, lists:zip(lists:seq(1, 10),Outputs)),

      io:format("Label: ~p, Predictions: ~p~n", [Label, Predictions]),
      Hit = case Label of
        0 ->
          {10,    lists:max(lists:map(fun ({_, P}) -> P end, Predictions))} =:= lists:keyfind(10,    1, Predictions);
        _ ->
          {Label, lists:max(lists:map(fun ({_, P}) -> P end, Predictions))} =:= lists:keyfind(Label, 1, Predictions)
      end,
      io:format("Hit ~p~n", [Hit]),
      Acc + case Hit of
              true -> 1;
              false -> 0
            end
    end,
    0,
    TrainingSet
  ).