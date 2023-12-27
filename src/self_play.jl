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
	
function update_dict(dict, arr::Vector{Tuple{String, Vector{Float64}, Float64}}, result::Integer)
    # https://www.gm.th-koeln.de/ciopwebpub/Kone15c.d/TR-TDgame_EN.pdf
    # for changing the game values at each state
    
    γ = 0.9
    alpha = 0.1
    for i in 1:size(arr)[1]
        pos = size(arr)[1] - i + 1
        if pos == size(arr)
            arr[pos][3] = result
        else
            delta = γ * arr[pos + 1][3] - arr[pos][3]
            arr[pos][3] += alpha * delta
        end

    for entry in arr
        if !(entry[1] in keys(dict))
            dict[entry[1]] = (spzeros(Int, 4096), 0, 0)
        end
        current_data = dict[entry[1]]
        new_data = (current_data[1], current_data[2] + 1, current_data[3] + result[3])
        new_data[1][move_int] += 1
        dict[entry[1]] = new_data
    end
    return dict
end


function self_play(model::ChessNet, arguments::Dict{String, Int}, positions_file::String)
	pos_dict = Dict{String, Tuple{SparseVector{Int, Int}, Int, Int}}()
    # load positions from most common positions
    positions = load_most_common(positions_file)
    
    game_num = 1
	for game_num in 1:arguments["num_games"]
        # play 100 games from common positions
        if game_num > 900
            game = training_self_game(model, pos_dict, positions[game_num], arguments)
            pos_dict =  update_dict(pos_dict, game)
        elseif game_num > 100
            # make 10 almost random moves (probabilities >= 0.1, if none are like this, purely random)
            board = startboard()
            for i in 1:20
                # make the kinda random mvoe
            end
            game = trainig_self_game(model, pos_dict, "", arguments)
        else
            board = startboard()
            for i in 1:10
                # make random move
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

function train_model(model::ChessNet, dict::Dict{String, Tuple{SparseVector{Int, Int}, Float32}})
    return model
end


if abspath(PROGRAM_FILE) == @__FILE__
	JLD2.@load "../models/supervised_model.jld2" model
    arguments = Dict{String, Int}()
    arguments["num_games"] = 1000
    arguments["num_searches"] = 100
    arguments["C"] = 2
    model = self_play(model, arguments, "../data/common_games.txt")
end
