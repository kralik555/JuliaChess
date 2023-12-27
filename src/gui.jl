using Mousetrap
using Chess

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
TS = 100

function update_chessboard(grid::Grid, board::Board)
    domoves!(board, rand(moves(board)))
    for i in 1:64
        col = (i - 1) % 8 + 1
        row = div(i - 1, 8) + 1
        if pieceon(board, Square(i)) != EMPTY
            piece = pieceon(board, Square(i))
            color = tochar(pcolor(piece))
            type = tochar(ptype(piece))
            image = ImageDisplay("piece_sprites/$(color)$(type).png")
            insert_at!(grid, image, row, col)
        else
            square = Label("")
            add_css_class!(square, "chess-square")
            if (col + row) % 2 == 0
                add_css_class!(square, "chess-square-white")
            else
                add_css_class!(square, "chess-square-black")
            end
            insert_at!(grid, square, row, col)
        end
    end
end

function create_chessboard(app)
    add_css!("""
        .chess-square {
            background-color: blue;  # Default background color
            color: white;            # Default text color
            border-radius: 0%;      # Rounded corners
        }
        .chess-square-black {
            background-color: gray;
            color: gray;
        }
        .chess-square-white {
            background-color: white;
            color: white;
        }
        """)
    
    window = Window(app)

    set_size_request!(window, Vector2f(8 * TS, 8 * TS))

    grid = Grid()
    set_size_request!(grid, Vector2f(8 * TS, 8 * TS))


    for row in 1:8
        for col in 1:8
            square = Label("$(8 * (row - 1) + col)")  # or Button()

            add_css_class!(square, "chess-square")
            if (row + col) % 2 == 0
                add_css_class!(square, "chess-square-white")
            else
                add_css_class!(square, "chess-square-black")
            end
            # Add the square to the grid at the correct position
            set_size_request!(square, Vector2f(TS, TS))
            insert_at!(grid, square, col, row, 1, 1)
        end
    end

    image = ImageDisplay("piece_sprites/bk.png")

    insert_at!(grid, image, 6, 4)
    set_child!(window, grid)
    board = startboard()
    update_chessboard(grid, board)   
    present!(window)
end

function main()
    app = Application("com.example.chessboard")
    connect_signal_activate!(app) do app
        create_chessboard(app)
    end

    run!(app)
end

main()
