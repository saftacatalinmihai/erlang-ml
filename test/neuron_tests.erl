-module(neuron_tests).

-include_lib("eunit/include/eunit.hrl").

perceive_test_() ->
    [   fun () ->
            neuron:perceive([neuron:perceive([1,1], [-30,20,20]), neuron:perceive([1,1], [30,-20,-20])], [-10,20,20]) > 0.9999
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