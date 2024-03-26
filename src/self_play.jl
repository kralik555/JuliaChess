using Chess
using SparseArrays
using Serialization
include("board_class.jl")
include("mcts.jl")
include("model.jl")
include("test.jl")
include("data_reader.jl")
include("supervised_training.jl")

function training_self_game(model::ChessNet, starting_position::String, args::Dict{String, Union{Float64, Int64}}, game_num)
    if starting_position == ""
		board = startboard()
	else
		board = fromfen(starting_position)
	end
    println(fen(board))
    arr = Vector{Tuple{String, SparseVector{Float64}, Float64}}()
    pos_arr = Vector{String}()
    is_repetition = false
    while !(isterminal(board))
        if fen(board) in pos_arr
            is_repetition = true
            break
        end
        time0 = time()
        (probs, move, value) = tree_move_with_distro(model, board, args)
		move = int_to_move(Int(only(move)))
        push!(arr, (fen(board), probs, only(value)))
        push!(pos_arr, fen(board))
        domove!(board, move)
        println(move, "\t", game_num)
	end
    result = 0
    if is_repetition == false
        result = game_result(board)
    else
        result = repetition_result(board)
    end
    println(result)
    return arr, result
end
	
function update_dict(dict::Dict{String, Tuple{SparseVector{Float64}, Int, Float64}}, arr::Vector{Tuple{String, SparseVector{Float64}, Float64}}, result::Float64)
    # https://www.gm.th-koeln.de/ciopwebpub/Kone15c.d/TR-TDgame_EN.pdf
    
    γ = 0.9
    alpha = 0.1
    # temporal difference
    for i in 1:size(arr)[1]
        pos = size(arr)[1] - i + 1
        if pos == size(arr)[1]
            arr[pos] = (arr[pos][1], arr[pos][2], result)
        else
            delta = γ * arr[pos + 1][3] - arr[pos][3]
            arr[pos] = (arr[pos][1], arr[pos][2], arr[pos][3] + alpha * delta)
        end
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


function self_play_training(model::ChessNet, arguments::Dict{String, Union{Float64, Int64}}, positions_file::String)
    # key = board FEN
    # Sparse vector = move probabilities based on the tree
    # Int = number of visits - to divide the final result by
    # Float64 is the value of the position (first from model, then changed by the temporal difference
    # load positions from most common positions
    positions = load_most_common(positions_file)
    try
        mkdir("temp")
    catch e
        rm("temp", recursive=true)
        mkdir("temp")
    end
    lk = ReentrantLock()
    Threads.@threads :dynamic for game_num in 1:arguments["num_games"]
        println("Game number ", game_num)
        # play 100 games from common positions
        if game_num > 900
            arr, result = training_self_game(model, positions[game_num - 900], arguments, game_num)
            #=while !trylock(lk)
                continue
            end=#
            pos_dict = update_dict(pos_dict, arr, result)
            #unlock(lk)
        elseif game_num > 100
            # make 10 almost random moves (probabilities >= 0.1, if none are like this, purely random)
            board = startboard()
            for i in 1:20
                (probs, _) = model_move_with_distro(model, board)
                move = int_to_move(rand(probs))
                domove!(board, move)
            end
            arr, result = trainig_self_game(model, fen(board), arguments, game_num)
            #=while !trylock(lk)
                continue
            end=#
            pos_dict = update_dict(pos_dict, arr, result)
            #unlock(lk)
        else
            board = startboard()
            for i in 1:10
                domove!(board, rand(moves(board)))
            end
            arr, result = training_self_game(model, fen(board), arguments, game_num)
            #=while !trylock(lk)
                continue
            end=#
            pos_dict = update_dict(pos_dict, arr, result)
            #unlock(lk)
        end
        serialize("temp/resut_arr_$(game_num).bin", (arr, result))
    end
    # train model on dict
    files = readdir("temp/")
    num_files = size(files)[1]
	pos_dict = Dict{String, Tuple{SparseVector{Float64}, Int, Float64}}()
    for i in 1:num_files
        arr, result = deserialize("temp/data_$i.0.bin")
        pos_dict = update_pos_dict(pos_dict, arr, result)
        if size(collect(keys(pos_dict))) >= 512
            model = train_model(model, pos_dict)
            pos_dict = Dict{String, Tuple{SparseVector{Float64}, Int, Float64}}()
        end
    end
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
    println("Trained, waiting for save")
    model_save_path = "../models/self_play_models/"
    num_models = size(readdir(model_save_path))[1]
    model_path = "$(model_save_path)sp_model_$(num_models + 1).jld2"
    JLD2.@save model_path model
    println("Saved model!")
    return model
end


if abspath(PROGRAM_FILE) == @__FILE__
	JLD2.@load "../models/supervised_model.jld2" model
    arguments = Dict{String, Union{Float64, Int64}}()
    arguments["num_games"] = 1000
    arguments["num_searches"] = 70
    arguments["C"] = 2
    arguments["search_time"] = 0.5
    sth, _ = model.model(board_to_tensor(startboard()))
    model = self_play_training(model, arguments, "../data/common_games.txt")
    println("Training finished!")
end
