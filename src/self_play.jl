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
        (probs, move, value) = tree_move_with_distro(model, board, args)
		move = int_to_move(Int(only(move)))
        push!(arr, (fen(board), probs, only(value)))
        push!(pos_arr, fen(board))
        domove!(board, move)
<<<<<<< HEAD
=======
        println(move, "\t", game_num)
>>>>>>> ee810fd (Added stockfish)
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


<<<<<<< HEAD
function self_play_training(models::Vector{ChessNet}, arguments::Dict{String, Float64}, positions_file::String)
=======
function self_play_training(model::ChessNet, arguments::Dict{String, Union{Float64, Int64}}, positions_file::String)
>>>>>>> ee810fd (Added stockfish)
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
<<<<<<< HEAD
    num_threads = Threads.nthreads()
    Threads.@threads :static for game_num in 1:arguments["num_games"]
        model = models[Threads.threadid()]
        println("Game number ", game_num)
        # play 100 games from common positions
        time0 = time()
        if game_num > 400
            arr, result = training_self_game(model, positions[game_num - 900], arguments)
=======
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
>>>>>>> ee810fd (Added stockfish)
        else
            board = startboard()
            for i in 1:10
                domove!(board, rand(moves(board)))
                if size(moves(board))[1] == 0
                    break
                end
            end
            if size(moves(board))[1] == 0
                continue
            end
<<<<<<< HEAD
            arr, result = training_self_game(model, fen(board), arguments)
        end
        println("Time: ", time() - time0)
        serialize("temp/data_$(Int64(game_num)).bin", (arr, result))
    end
    # train model on dict
    println("Finished games!")
    model = training_on_games(models[1])
=======
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
>>>>>>> ee810fd (Added stockfish)
    return model
end

function training_on_games(model::ChessNet)
    files = readdir("temp")
    num_files = size(files)[1]
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
    num_models = size(readdir("../models/self_play_models/"))[1]
    JLD2.@save ("../models/self_play_models/model_$(num_models + 1).jld2") model
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


if abspath(PROGRAM_FILE) == @__FILE__
<<<<<<< HEAD
    try
        files = readdir("temp")
        model_num = size(readdir("../models/self_play_models/"))[1]
        JLD2.@load "../models/self_play_models/model_$(model_num).jld2" model
        model = training_on_games(model)
    catch e
    end
    for i in 1:5
        models = Vector{ChessNet}()
        nthreads = Threads.nthreads()
        for i in 1:nthreads
            saved_models = readdir("../models/self_play_models")
            num_models = size(saved_models)[1]
            if num_models == 0
                JLD2.@load "../models/supervised_model_1.jld2" model
                push!(models, model)
            else
                JLD2.@load "../models/self_play_models/model_$(num_models).jld2" model
                push!(models, model)
            end
        end
        arguments = Dict{String, Float64}()
        arguments["num_games"] = 200
        arguments["num_searches"] = 100
        arguments["C"] = 2
        arguments["search_time"] = 0.8
        Threads.@threads for i in 1:nthreads
            n, _ = models[Threads.threadid()].model(board_to_tensor(startboard()))
        end
        time0 = time()
        model = self_play_training(models, arguments, "../data/common_games.txt")
        println("Time for training: ", time() - time0)
    end
=======
	JLD2.@load "../models/supervised_model.jld2" model
    arguments = Dict{String, Union{Float64, Int64}}()
    arguments["num_games"] = 1000
    arguments["num_searches"] = 70
    arguments["C"] = 2
    arguments["search_time"] = 0.5
    sth, _ = model.model(board_to_tensor(startboard()))
    model = self_play_training(model, arguments, "../data/common_games.txt")
    println("Training finished!")
>>>>>>> ee810fd (Added stockfish)
end
