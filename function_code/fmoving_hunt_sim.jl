using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro

# Creating Wolf agent type
@agent struct Wolf(ContinuousAgent{2, Float64})
end

# Creating Sheep agent type
@agent struct Sheep(ContinuousAgent{2, Float64})
end

# Function to move a sheep at each time step, assuming it moves in a circular path with decreasing velocity
function animal_step!(agent::Sheep, model)

    center = model.center # center of the sheep's circular path (arbitrarily chosen by user)
    dt = model.dt # time step increment for our sim
    sheep_tangent_acceleration = model.sheep_tangent_acceleration # accel in tangent dir to sheep's circular movement

    # Find unit vector pointing from sheep to center (radial direction)
    radius = norm(agent.pos - center)
    unit_radial_vector = (center - agent.pos) / radius

    # Find the centripetal acceleration needed for the sheep to follow its circular path
    new_speed = norm(agent.vel) + sheep_tangent_acceleration * dt
    centrip_accel = (new_speed^2/radius)* unit_radial_vector

    agent.vel += centrip_accel * dt

    # Project the sheep's velocity onto the tangential direction (obtained by rotating the radial 
    # vector 90° counterclockwise) to maintain circular motion
    rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
    tangential_vector = rotation_matrix * unit_radial_vector
    proj = tangential_vector * dot(agent.vel, tangential_vector)/norm(tangential_vector)
    agent.vel = proj/norm(proj) * new_speed

    # update model with new velocity after dt (timestep increment for simulation)
    move_agent!(agent, model, dt) 

end


# Function to move a wolf at each time step, assuming prey moves in a circular path with decreasing velocity
function animal_step!(agent::Wolf, model)

    # identify the sheep in the model
    sheep = nothing
    for a in allagents(model)
        if typeof(a) == Sheep
            sheep = a
            break                       # exit loop once prey is found
        end
    end

    # Model the behavior of the wolf

    current_distance = norm(sheep.pos - agent.pos)

    wolf_chase_speed = model.wolf_chase_speed # speed at which wolf chases sheep
    min_safe_distance = model.min_safe_distance # minimum distance at which wolf will orbit sheep
    ww_force_coefficient = model.ww_force_coefficient # repulsive force coefficient for wolf-wolf interaction
    dt = model.dt

    # each neighbor wolf exerts repulsive force on current wolf agent
    wolf_repulsion = [0, 0] # stores net wolf-wolf repulsive force vector on current wolf
    for neighbor in allagents(model)

        if neighbor.id != agent.id && isa(neighbor, Wolf)
            
            # sum up wolf-wolf repulsive force vectors
            # repulsive force is inversely proportional to distance between wolves
            distance_between = norm(agent.pos - neighbor.pos)
            wolf_repulsion += ww_force_coefficient * (agent.pos - neighbor.pos) / (distance_between)^2

        end

    end
    
    # once wolf is within a critical distance to the sheep, wolf will maintain that critical distance
    # and orbit around sheep
    if (current_distance <= min_safe_distance)

        # find direction of wolf to sheep, rotate 90 degrees to find tangential movement direction
        rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
        u = rotation_matrix * (agent.pos - sheep.pos)

        # projecting the repulsive force vector onto the tangential vector and multiply by wolf speed
        # to determine the velocity the wolf travels along the circle
        dot_product = dot(u, wolf_repulsion)
        proj_u_v = (dot_product / norm(u)^2) * u * wolf_chase_speed
        agent.vel = proj_u_v

    # if wolf is outside critical distance from sheep, wolf is attracted to sheep
    else

        # wolf moves in direction of sheep due to attractive force exerted by sheep on wolf
        wolf_chase_velocity = (sheep.pos - agent.pos) / norm(sheep.pos - agent.pos) * wolf_chase_speed

        # wolf velocity impacted by both attraction to sheep and repulsion to neighboring wolves
        agent.vel = wolf_chase_velocity + wolf_repulsion
    
    end
    
    # update model with new velocity after dt (timestep increment for simulation)
    move_agent!(agent, model, dt)
end

# this fn initializes agent-based model for wolf hunt of a single prey moving in circular path w/ decreasing velocity
function initialize(total_agents, size, min_safe_distance, ww_force_coefficient, sw_force_coefficient,
                   wolf_chase_speed, dt, seed,
                   center, sheep_tangent_acceleration, sheep_init_speed)
    space = ContinuousSpace(size; periodic = false)
    properties = Dict(
        :min_safe_distance => min_safe_distance,
        :ww_force_coefficient => ww_force_coefficient,
        :sw_force_coefficient => sw_force_coefficient,
        :wolf_chase_speed => wolf_chase_speed,
        :dt => dt,
        :center => center,
        :sheep_tangent_acceleration => sheep_tangent_acceleration
    )
    rng = Xoshiro(seed)

    model = StandardABM(Union{Wolf, Sheep}, space; properties = properties, agent_step! = animal_step!, rng,
        container = Vector, # agents are not removed, so we use this
        scheduler = Schedulers.Randomly() # all agents are activated once at random
    )

    rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
    rand_pos = [5 + 10 *rand(rng), 5 + 10*rand(rng)]
    radial_vector = rand_pos - center
    tangential_vector = rotation_matrix * radial_vector
    initial_vel = (tangential_vector/norm(tangential_vector)) * sheep_init_speed

    # adding wolf agents at random positions with velocity = 0
    for n in 1:(total_agents - 1)
        add_agent!(Wolf, model; vel = (0.0, 0.0))
    end 

    # adding a single sheep/prey agent at random position with velocity = 0
    add_agent!(Sheep, model; pos = rand_pos, vel = initial_vel) # add one sheep
    return model
end

# this fn returns a video sim of a wolf pack hunting a single prey moving in circular path w/ decreasing velocity
function fmoving_hunt_sim(min_safe_distance=1.0, ww_force_coefficient=0.5, sw_force_coefficient=2.0, 
                          wolf_chase_speed=1.0, dt=0.25, 
                          total_agents=6, size=(20.0, 20.0), seed=125, framerate=15, frames=200,
                          center=[10.0, 10.0], sheep_tangent_acceleration=-0.1, sheep_init_speed=0.5)

    model = initialize(
        total_agents, size, min_safe_distance, ww_force_coefficient, sw_force_coefficient,
        wolf_chase_speed, dt, seed,
        center, sheep_tangent_acceleration, sheep_init_speed
    )

   # Define color, size, and marker functions
    ac(a::Wolf) = :green
    ac(a::Sheep) = :blue
    as(a::Wolf) = 10
    as(a::Sheep) = 13
    am = 'o'

    # Create the animation
    abmvideo("wolf_hunt.mp4", model;
    title = "Moving Sheep Hunt Simulation", framerate, frames, ac, as, am)

end

# test with 5 wolves and 1 sheep
fmoving_hunt_sim(
    1.0,                    # min_safe_distance
    0.5,                    # ww_force_coefficient
    2.0,                    # sw_force_coefficient
    1.0,                    # wolf_chase_speed (hardcoded inside animal_step! in 2nd version)
    0.25,                   # dt
    6,                      # total_agents (5 wolves + 1 sheep)
    (20.0, 20.0),           # size of the space
    125,                    # seed
    15,                     # framerate
    200,                    # frames
    [10.0, 10.0],           # center of circular sheep path
    -0.1,                   # sheep_tangent_acceleration
    0.5                     # sheep_init_speed
)