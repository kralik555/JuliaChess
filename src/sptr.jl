using Chess
using SparseArrays
using Serialization

function play()
    try
        mkdir("temp")
    catch e 
        rm("temp", recursive=true)
        mkdir("temp")
    end
    lk = ReentrantLock()
    Threads.@threads for i in 1:100
        if i > 50
            board = startboard()
            for j in 1:6
                move = rand(moves(board))
                domove!(board, move)
            end
                game = SimpleGame(board)
                while !isterminal(game)
                    domove!(game, rand(moves(game.board)))
                    print(i, " ")
                    sleep(rand() / 10)
                end
        else
            board = startboard()
            game = SimpleGame(board)
            while !(isterminal(game))
                domove!(game, rand(moves(game.board)))
                print(i, " ")
                sleep(rand() / 10)
            end
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    play()
end
