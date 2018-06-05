#Neural net in julia
module mlp
#setting parameters
n_hidden = nothing
lr = nothing
max_iter = nothing
w0 = nothing
w1 = nothing
error_vector = nothing

function set(n_hidden_=3, lr_=0.4, max_iter_=500, seed=nothing)
    global n_hidden = n_hidden_
    global lr = lr_
    global max_iter = max_iter_
    if seed!=nothing
        srand(seed)
    end
end
#defining derivative, using tanh as tf
derivative(x) = 1 - (tanh.(x).^2)

#defining feedforward part
function feedforward(X)
    global w0
    global w1
    #println( size(X), size(w0), size(w1))

    hidden = tanh.(X*w0)
    #println("Hidden shape", size(hidden))
    output = tanh.(hidden*w1)
    #println("Output shape", size(output))
    return [hidden, output]
end

#defining the predict function
predict(X) = feedforward(X)[2]
#shape corrector
function y_reshaper(y)
    if size(y)[2] > size(y)[1]
        return y'
    else
        return y
    end
end

#training
function fit!(X, y)

    y = y_reshaper(y)
    global w0 = randn(size(X)[2], n_hidden)
    global w1 = randn(n_hidden, size(y)[2])
    #println(size(w0))
    #println(size(w1))
    global max_iter
    global n_hidden
    # training
    #error vector
    global error_vector = ones(max_iter)
    println("starting training")
    for i = 1:max_iter

        h, o = feedforward(X)
        o_error = y - o
        o_delta = lr * o_error .* derivative(o)

        # propagating error
        h_error = o_delta * w1'
        h_delta = lr * h_error .* derivative(h)

        err = sum(abs.(o_error))
        global error_vector[i] = err
        if (i%200)==0
            println("Accumulated error: ", err)
        end

        #updating weights
        global w0 += (X' * h_delta)
        global w1 += (h' * o_delta)
    end
    println("OVER")
end


end

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
using mlp
using PyPlot

X = [0 0; 0 1; 1 0; 1 1]
y = [0 1 1 0]

mlp.set(3, 0.4, 500, 2)
mlp.fit!(X,y)

println("\n\n%%%%%%%results%%%%%%%")
println("Total iterations: ", mlp.max_iter, "| Learning rate: ", mlp.lr, "| Nodes in the hidden: ",mlp. n_hidden)
results = mlp.predict(X)
#println(results)
println("Final accumulated error: ", sum(abs.(y'-results)))
for i=1:4
    println(X[i,:], " : ", results[i, :])
end
#plot

plot(mlp.error_vector)
title("Error x iteration")
ylabel("Accumalated error")
xlabel("Iteartion")
show()
