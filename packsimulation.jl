using Agents

size = (1.0, 1.0)
space = ContinuousSpace(size; periodic = false;)

@agent struct Wolf(ContinuousAgent{2, Float64})
    group::Int
end