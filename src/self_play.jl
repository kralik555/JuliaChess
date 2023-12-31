using Chess
using SparseArrays
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
    arr = Vector{Tuple{String, Vector{Float64}, Float64}}()
    while !(isterminal(game))
        (probs, move) = tree_move_with_distro(model, game, args)
        _, value = model.model(board_to_tensor(game.board))
		move = int_to_move(Int(only(move)))
        println(tostring(move))
        push!(arr, (fen(game.board), probs, value))
        domove!(game, move)
	end
    res = game.headers.:result
    result = res == "1-0" ? 1 : res == "0-1" ? -1 : 0
    
    return arr, result
end
	
function update_dict(dict::Dict{String, Tuple{SparseVector{Float64}, Int, Float64}}, arr::Vector{Tuple{String, SparseVector{Float64}, Float64}}, result::Integer)
    # https://www.gm.th-koeln.de/ciopwebpub/Kone15c.d/TR-TDgame_EN.pdf
    
    γ = 0.9
    alpha = 0.1
    # temporal difference
    for i in 1:size(arr)[1]
        pos = size(arr)[1] - i + 1
        if pos == size(arr)
            arr[pos][3] = result
        else
            delta = γ * arr[pos + 1][3] - arr[pos][3]
            arr[pos][3] += alpha * delta
        end
    end

    dict_keys = collect(key(dict))
    for entry in arr
        if !(entry[1] in dict_keys)
            dict[entry[1]] = (spzeros(Int, 4096), 0, 0)
        end
        current_data = dict[entry[1]]
        new_data = Tuple{SparseVector{Float64}, Int, Float64}(current_data[1], current_data[2] + 1, current_data[3] + result)
        new_data[1][move_int] += 1
        dict[entry[1]] = new_data
    end
    return dict
end


function self_play_training(model::ChessNet, arguments::Dict{String, Float64}, positions_file::String)
    # key = board FEN
    # Sparse vector = move probabilities based on the tree
    # Int = number of visits - to divide the final result by
    # Float64 is the value of the position (first from model, then changed by the temporal difference
	pos_dict = Dict{String, Tuple{SparseVector{Float64}, Int, Float64}}()
    # load positions from most common positions
    positions = load_most_common(positions_file)
    
    game_num = 1
	for game_num in 1:arguments["num_games"]
        println("Game number ", game_num)
        # play 100 games from common positions
        if game_num > 900
            arr, result = training_self_game(model, pos_dict, positions[game_num], arguments)
            pos_dict =  update_dict(pos_dict, arr, result)
        elseif game_num > 100
            # make 10 almost random moves (probabilities >= 0.1, if none are like this, purely random)
            board = startboard()
            for i in 1:20
                probs = model_move_distro(model, board)
                move = int_to_move(rand(probs))
                domove!(board, move)
            end
            arr, result = trainig_self_game(model, pos_dict, "", arguments)
            pos_dict = update_dict(pos_dict, arr, result)
        else
            board = startboard()
            for i in 1:10
                domove!(board, rand(moves(board)))
            end
            arr, result = training_self_game(model, board.fen, arguments)
            pos_dict = update_dict(pos_dict, arr, result)
        end
        game_num += 1
    end
    # change dict
    # train model on dict
    model = train_model(model, pos_dict)
    return model
end


function train_model(model::ChessNet, dict::Dict{String, Tuple{SparseVector{Float64}, Int, Float32}})
    return model
end


if abspath(PROGRAM_FILE) == @__FILE__
	JLD2.@load "../models/supervised_model.jld2" model
    arguments = Dict{String, Float64}()
    arguments["num_games"] = 1000
    arguments["num_searches"] = 100
    arguments["C"] = 2
    model = self_play_training(model, arguments, "../data/common_games.txt")
end
