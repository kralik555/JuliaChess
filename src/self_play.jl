using LinearAlgebra
using MKL
using Chess
using SparseArrays
using Serialization
using StatsBase
using Flux.Data: DataLoader
using JLD2
include("board_class.jl")
include("mcts.jl")
include("model.jl")
include("test.jl")
include("data_reader.jl")

function training_self_game(model::ChessNet, starting_position::String, args::Dict{String, Float64})
    if starting_position == ""
		board = startboard()
	else
		board = fromfen(starting_position)
	end
    game = SimpleGame(board)
    arr = Vector{Tuple{String, SparseVector{Float64}, Float64}}()
    pos_arr = Vector{String}()
    num_moves = 0
    while !(isterminal(game))
        if fen(game.board) in pos_arr
            break
        elseif rand() < 0.1
            probs, _, value = tree_move(model, game, args)
            push!(arr, (fen(game.board), probs, only(value)))
            push!(pos_arr, fen(game.board))
            move = rand(moves(game.board))
            println(move, " was randomly chosen")
            domove!(game, move)
            continue
        end
        probs, move, value = tree_move(model, game, args)
		move = int_to_move(Int(only(move)))
        push!(arr, (fen(game.board), probs, only(value)))
        push!(pos_arr, fen(game.board))
        println(move)
        domove!(game, move)
        if num_moves >= 100
            break
        end
	end
    result = game_result(game)
    println(result)
    return arr, result
end
	
function update_dict(dict::Dict{String, Tuple{SparseVector{Float64}, Int, Float64}}, arr::Vector{Tuple{String, SparseVector{Float64}, Float64}}, result::Float64)
    gamma = 0.9
    # temporal difference
    arr[size(arr)[1]] = result
    for i in 1:size(arr)[1] - 1
        pos = size(arr)[1] - i
        delta = gamma * arr[pos + 1][3] - arr[pos][3]
        arr[pos] = (arr[pos][1], arr[pos][2], arr[pos][3] + delta)
    end

    dict_keys = collect(keys(dict))
    for entry in arr
        if !(entry[1] in dict_keys)
            dict[entry[1]] = (spzeros(Int, 4096), 0, 0)
        end
        current_data = dict[entry[1]]
        new_data = (current_data[1], current_data[2] + 1, current_data[3] + result)
        dict[entry[1]] = new_data
    end
    return dict
end


function self_play_training(model::ChessNet, arguments::Dict{String, Float64}, positions_file::String)
    positions = load_most_common(positions_file)
    try
        mkdir("temp")
    catch e
    end
   for game_num in 1:200
        if game_num > 100
            arr, result = training_self_game(model, positions[game_num - 100], arguments)
        elseif game_num == 1
            arr, result = training_self_game(model, fen(startboard()), arguments)
        else
            board = startboard()
            for i in 1:10
                domove!(board, rand(moves(board)))
                if size(moves(board))[1] == 0
                    break
                end
            end
            arr, result = training_self_game(model, fen(board), arguments)
        end
        serialize("temp/data_$(Int64(game_num)).bin", (arr, result))
        if game_num % 10 == 0
            model = train_on_games(model)
            mkdir("temp")
        end
    end
    # train model on dict
    println("Finished games!")
    model = training_on_games(model)
    return model
end

function training_on_games(model::ChessNet, opt=Adam(0.001))
    files = readdir("temp")
    pos_dict = Dict{String, Tuple{SparseVector{Float64}, Int, Float64}}()
    for i in 1:200
        if !("data_$i.bin" in files)
            continue
        end
        arr, result = deserialize("temp/data_$i.bin")
        pos_dict = update_dict(pos_dict, arr, result)
        if size(collect(keys(pos_dict)))[1] >= 256
            model = train_model(model, pos_dict)
            pos_dict = Dict{String, Tuple{SparseVector{Float64}, Int, Float64}}()
        end
    end
    if size(collect(keys(pos_dict)))[1] > 0
        model = train_model(model, pos_dict)
    end
    num_models = size(readdir("../models/"))[1]
    JLD2.@save ("../models/self_play_model_$(num_models + 1).jld2") model
    rm("temp", recursive=true)
    return model
end

function train_model(model::ChessNet, dict::Dict{String, Tuple{SparseVector{Float64}, Int, Float64}})
    pos_keys = collect(keys(dict))
    tensors = []
    move_distros = []
    game_values = []
    for key in pos_keys
        board = fromfen(key)
        tensor = create_input_tensors(board)
        move_distr = Vector(dict[key][1])
        value = dict[key][3] / dict[key][2]
        push!(tensors, tensor)
        push!(move_distros, reshape(move_distr, 1, :))
        push!(game_values, value)
    end
    tensors = permutedims(cat(tensors..., dims=4), (2, 3, 1, 4))
    move_distros = vcat(move_distros...)
    model = train_batch(model, tensors, move_distros, game_values)
    println("Trained one batch")
    return model
end

function train_batch_sp(model, tensors, policies, values, opt)
	function loss(x, y_moves, y_value)
		y_pred_moves, y_pred_value = model.model(x)
 		move_loss = Flux.crossentropy(y_pred_moves, y_moves)

        y_pred_value = dropdims(y_pred_value; dims=1)
        value_loss = Flux.mse(y_pred_value, y_value)
        return move_loss + 40 * value_loss
	end

    data_loader = DataLoader((tensors, policies, values), batchsize=length(values), shuffle=true)
	
	for (x_batch, y_move_batch, y_value_batch) in data_loader
		Flux.train!(loss, Flux.params(model.model), [(x_batch, y_move_batch, y_value_batch)], opt)
	end
    return model
end

function train_model_no_tree_games(model, states, correct_policies, correct_values, opt)
    tensors = []
    for f in states
        push!(tensors, board_to_tensor(fromfen(f)))
    end
    tensors = permutedims(cat(tensors..., dims=4), (1, 2, 3, 4))
    model = train_batch_sp(model, tensors, correct_policies, correct_values, opt)
    return model
end

function change_arrays(states, moves, policies, values, result)
    new_policies = Vector{Vector{Float32}}()
    new_values = copy(values)
    l = length(values)
    gamma = 0.9
    new_values[l] = result
    for i in 1:(l - 1)
        diff = new_values[l - i + 1] - new_values[l - i]
        new_values[l - i] += gamma * diff
    end
    turn = 1
    for i in 1:length(moves)
        policy = Vector(policies[i])
        adj = 1
        if turn == result
            adjustment = 1.02
        elseif turn == -result
            adjustment = 0.98
        else
            adjustment = 0.99
        end

        policy[moves[i]] *= adjustment
        policy /= sum(policy)
        push!(new_policies, policy)
        
        if turn == 1
            turn = -1
        else
            turn = 1
        end
    end
    new_policies = hcat(new_policies...)
    return new_policies, new_values
end

function self_play_no_tree(model::ChessNet, opt)
    for game_num in 1:100
        states = Vector{String}()
        played_moves = Vector{Integer}()
        policies = Vector{SparseVector{Float32}}()
        values = Vector{Float32}()
        println("Game ", game_num)
        board = startboard()
        game = SimpleGame(board)
        while !isterminal(game)
            if fen(game.board) in states
                move = rand(moves(game.board))
                domove!(game, move)
                continue
            end
            can_end = false
            if !haslegalmoves(game.board)
                println("No legal moves, but game not ended!")
                break
            end
            for m in moves(game.board)
                try
                    undoinfo = domove!(game.board, m)
                    if ischeckmate(game)
                        can_end = true
                        break
                    end
                    undomove!(game.board, undoinfo)
                catch
                    println("An error occured!")
                    can_end = true
                end
            end
            if can_end
                break
            end
            policy, value = model.model(board_to_tensor(board))
            policy = policy[1]
            policy *= get_valid_moves(game.board)
            move = sample(1:4096, ProbabilityWeights(policy))
            value = only(value)
            push!(values, value)
            push!(states, fen(game.board))
            push!(played_moves, move)
            push!(policies, SparseVector(policy))
            played_move = int_to_move(move)
            domove!(game, played_move)
        end
        result = game_result(game)
        println(result)
        new_policies, new_values = change_arrays(states, played_moves, policies, values, result)
        model = train_model_no_tree_games(model, states, new_policies, new_values, opt)
        num_models = size(readdir("../models/"))[1]
        JLD2.@save "../models/self_play_model_$(num_models + 1).jld2" model
    end
    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = ChessNet()
    JLD2.@load "../models/final_model.jld2" model
    opt = Adam(0.0001)
    for epoch in 1:1000
        println("Epoch: ", epoch)
        self_play_no_tree(model, opt)
    end
end
