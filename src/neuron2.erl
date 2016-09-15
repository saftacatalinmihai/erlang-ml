%%%-------------------------------------------------------------------
%%% @author mihai
%%% @copyright (C) 2016, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 15. Sep 2016 5:09 PM
%%%-------------------------------------------------------------------
-module(neuron2).
-author("mihai").

-compile(export_all).
%% API

-record(state, {
  inputs,
  outputs,
  inputs_received,
  backprops_received,
  bias,
  weights,
  learning_rate,
  batch_size,
  activation_fn,
  deriv_fn
}).

start() ->
  Pid = spawn(?MODULE, loop, [#state{}]),
  {ok, Pid}.

%%#state{
%%  inputs = Inputs,
%%  outputs = Outputs,
%%  inputs_received = InRec,
%%  backprops_received = BpRec,
%%  bias = Bias,
%%  weights = Weights,
%%  learning_rate = LR,
%%  batch_size = BastchSize,
%%  activation_fn = ActFn,
%%  deriv_fn = DerivFn
%%}

% Loop
loop(State) ->
  receive
    {FromPid, {forward, {Value, ExpectedList}}} ->
      NewInputs = lists:keyreplace(FromPid, 1, State#state.inputs, {FromPid, Value}),
      case all_inputs_received(State#state{inputs = NewInputs}) of
        true ->
          case last_layer(State) of
            true ->
              on_forward_last_layer(State#state{inputs = NewInputs}, ExpectedList);

            false ->
              on_forward_hidden_layer(State#state{inputs = NewInputs}, ExpectedList)

          end;
        false ->
          {noreply, State#state{inputs = NewInputs, inputs_received = State#state.inputs_received + 1}}
      end;
      io:format("Received msg: ~p~n", [X]),
      From ! {self(), {backprop, X}},
      loop(State)
  end.

send_all(Pids, Msg) ->
  RefPids = lists:map(fun(P) -> {P, make_ref()} end, Pids),
  [P ! {self(), Msg} || {P, _} <- RefPids],
  wait_responses(Pids).

wait_responses(Pids) -> wait_responses(Pids, []).
wait_responses([], Acc) -> Acc;
wait_responses(Pids, Acc) ->
  receive
    {From, Msg} -> wait_responses(lists:delete(From, Pids), [Msg | Acc])
  end.

test() ->
  {ok, P1} = neuron_node:start(),
  {ok, P2} = neuron_node:start(),
  {ok, P3} = neuron_node:start(),

  send_all([P1, P2, P3], hi).


all_inputs_received(State) ->
  State#state.inputs_received =:= length(State#state.inputs) - 1.

all_backprops_received(State) ->
  State#state.backprops_received =:= length(State#state.outputs) - 1.

last_layer(State) ->
  length(State#state.outputs) =:= 0.
