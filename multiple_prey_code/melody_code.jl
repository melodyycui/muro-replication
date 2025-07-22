using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro

# wolf agent type
@agent struct Wolf(ContinuousAgent{2, Float64})
end

# wolf agent type
@agent struct Sheep(ContinuousAgent{2, Float64})
    health::Float64
    speed::Float64
end

function safe_norm(v)
    return norm(v) == 0 ? [0.0, 0.0] : v / norm(v)
end

# fn updates the position and velocity of a sheep agent after one time step, dt
function animal_step!(agent::Sheep, model)

    sheep_mass = model.sheep_mass
    
    d_rep = model.d_rep # min. distance at which neighboring sheep start repelling each other

    # number of neighboring sheep exerting interaction forces on agent
    n_att = model.n_att # number of neighboring sheep the agent is attracted to
    n_ali = model.n_ali # number of neighboring sheep the agent aligns with
    n_rep = 0 # counts the number of neighboring sheep within drep of the agent

    # weight parameters
    w_rep_ws = model.w_rep_ws # for repulsive force from wolf
    w_rep_ss = model.w_rep_ss # for repulsive force from other sheep
    w_att = model.w_att # for attractive force from other sheep
    w_ali = model.w_ali # for alignment with other sheep

    # sheep agent wants to move away from wolf agents
    wolf_repulsion = [0, 0] # stores net repulsive force from wolves on sheep
    
    # each wolf exerts a repulsive force on the sheep agent
    for wolf in allagents(model)

        if isa(wolf, Wolf)

            # constant to control how fast force exponentially decays
            c = model.c
            
            # sum up the repulsive forces exerted by each wolf on the sheep
            # force is proportional to distance between wolf and sheep
            distance_between = norm(agent.pos - wolf.pos)
            # get the direction of the repulsive force
            unit_vector = (agent.pos - wolf.pos) / (distance_between)

            # force equation is a decaying exponential
            wolf_repulsion += (exp(-c * distance_between)) * unit_vector

        end

    end

    # sheep agent also experiences external interaction forces from neighboring sheep
    
    sheep_attraction = [0.0, 0.0] # will hold the sum attractive force from neighboring sheep
    sheep_repulsion = [0.0, 0.0] # will hold the sum repulsive force from neighboring sheep
    heading_angle = 0.0 # will hold average alignment angle

    # sheep repulsed by nearby sheep that are too close
    for sheep in allagents(model)

        # find all sheep within distance drep from sheep agent
        if isa(sheep, Sheep) && sheep.id != agent.id
            if norm(sheep.pos - agent.pos) < d_rep

                # add normalized repulsive force vector to total sheep repulsion
                sheep_repulsion += safe_norm(agent.pos - sheep.pos)
                # keep count of total number of sheep that are too close
                n_rep += 1
            end
        end
    end

    # collect all nearby sheep into an array
    neighbors = [i for i in nearby_agents(agent, model, 5.0)]

    # sheep agent attracted to n_att number of neighboring sheep
    # (how to prevent repeated random selection?)
    for i in 1:n_att

        if (length(neighbors) > 0)
            # choose a random neighbor to be attracted to and add attractive force to total
            neighbor = neighbors[rand(1:length(neighbors))]
            sheep_attraction += safe_norm(neighbor.pos - agent.pos)
        end

    end

    # sheep agent aligning with n_ali number of neighboring sheep
    for i in 1:n_ali

        if (length(neighbors) > 0)
            # ultimately have to make an array from the n_att sheep (rn, all of nearby sheep)
            # chose a random neighbor to align with and add alignment force to total
            neighbor = neighbors[rand(1:length(neighbors))]

            phi = atan(neighbor.vel[1], neighbor.vel[2])
            phi = phi < 0 ? phi + 2*pi : phi
            heading_angle += (1/n_ali) * phi

        end
        
    end

    # normalize all forces acting on sheep agent (so all vectors have magnitude = 1)
    wolf_repulsion = safe_norm(wolf_repulsion)
    sheep_repulsion = safe_norm(sheep_repulsion)
    sheep_attraction = safe_norm(sheep_attraction)
    sheep_alignment = [cos(heading_angle), sin(heading_angle)]

    # scale forces by weight parameters and sum together
    net_force = (w_rep_ws * wolf_repulsion) + (w_rep_ss * sheep_repulsion) + 
                (w_att * sheep_attraction) + (w_ali * sheep_alignment)
    
    # v = v0 + at, a = F/m => v = v0 + F/m * t
    agent.vel += (net_force/sheep_mass)*model.dt

    # direction of velocity is updated, but keep speed the same
    agent.vel = agent.health * agent.speed * safe_norm(agent.vel)

    if (agent.health >= 0.001)
        agent.health -= 0.001
    end

    # update agent
    move_agent!(agent, model, model.dt)

end

# fn updates the position and velocity of a wolf agent after one time step, dt
function animal_step!(agent::Wolf, model)

    wolf_mass = model.wolf_mass
    w_rep_ww = model.w_rep_ww
    circ_dist = model.circ_dist
    wolf_speed = model.wolf_speed
    dt = model.dt

    wolf_repulsion = [0, 0] # stores net wolf-wolf repulsive force vector on current wolf

    sheep_attraction = [0, 0]
    circling = false

    for sheep in allagents(model)

        if isa(sheep, Sheep)

            current_distance = norm(sheep.pos - agent.pos)

            # once wolf is within a critical distance to the sheep, wolf will maintain that critical distance
            # and orbit around sheep due to repulsion from other wolves
            if (current_distance <= circ_dist)

                # each neighbor wolf exerts repulsive force on current wolf agent
                for neighbor in allagents(model)

                    if neighbor.id != agent.id && isa(neighbor, Wolf)
                        
                        # repulsive force is proportional to 1/dist
                        dist_btwn = norm(agent.pos - neighbor.pos)
                        wolf_repulsion += w_rep_ww * (agent.pos - neighbor.pos) / (dist_btwn)^2

                    end

                end

                wolf_speed = 1.3 * model.wolf_speed

                # find direction of wolf to sheep, rotate 90 degrees to find tangential movement direction
                rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
                u = rotation_matrix * (agent.pos - sheep.pos)
                # projecting the repulsive force vector onto the tangential vector and multiply by wolf speed
                # to determine the velocity the wolf travels along the circle
                dot_product = dot(u, wolf_repulsion)
                proj_u_v = (dot_product / norm(u)^2) * u * wolf_speed
                agent.vel = proj_u_v
                circling = true
                break

            else

                # constants to control how fast force exponentially decays
                a = model.a
                b = model.b

                # summing attractive forces from sheep
                unit_vector = (sheep.pos - agent.pos) / current_distance
                sheep_attraction += a * exp(-b * current_distance) * unit_vector
            end
        end
    end

    if (!circling)
        
        agent.vel = safe_norm(sheep_attraction) * wolf_speed

        # acceleration = (sheep_attraction) / wolf_mass
        # agent.vel += acceleration * dt
        # agent.vel = wolf_speed * safe_norm(agent.vel)
    
    end
    
    # update model with new velocity after dt (timestep increment for simulation)
    move_agent!(agent, model, model.dt)
end

# this fn initializes our agent-based model for wolf hunt of a single reactive/escaping prey
function initialize(; size,
                    total_wolf, total_sheep,
                    circ_dist, d_rep, w_rep_ww,
                    a, b, c,
                    wolf_speed, sheep_speed,
                    wolf_mass, sheep_mass,
                    n_att, n_ali,
                    w_rep_ws, w_rep_ss, w_att, w_ali,
                    dt, seed)
    space = ContinuousSpace(size; periodic = false)

    properties = Dict{Symbol, Any}(
    :total_wolf => total_wolf,
    :total_sheep => total_sheep,
    :circ_dist => circ_dist,
    :d_rep => d_rep,
    :w_rep_ww => w_rep_ww,
    :a => a,
    :b => b,
    :c => c,
    :wolf_speed => wolf_speed,
    :sheep_speed => sheep_speed,
    :wolf_mass => wolf_mass,
    :sheep_mass => sheep_mass,
    :n_att => n_att, 
    :n_ali => n_ali,
    :w_rep_ws => w_rep_ws,
    :w_rep_ss => w_rep_ss,
    :w_att => w_att,
    :w_ali => w_ali,
    :dt => dt,
    :in_range => false
)

    rng = Xoshiro(seed)

    model = StandardABM(Union{Wolf, Sheep}, space;
                        properties=properties,
                        agent_step! = animal_step!,
                        rng=rng,
                        container=Vector,
                        scheduler=Schedulers.Randomly())

    # adding wolf agents at random positions in top-left corner with velocity = 0
    for n in 1:(total_wolf)

        rand_pos = [5*rand(rng), size[2] - 5*rand(rng)]
        add_agent!(Wolf, model; pos = rand_pos, vel = (0.0, 0.0))
    end

    # adding sheep agents around the center with velocity = 0
    for n in 1:(total_sheep) - 1
        
        rand_pos = [20 + 5*rand(rng), size[2] - 20 - 5 * rand(rng)]
        add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0), speed = sheep_speed, health = 1.0)

    end 

    # add a slower sheep
    rand_pos = [20 + 5*rand(rng), size[2] - 20 - 5 * rand(rng)]
    add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0), speed = 0.5*sheep_speed, health = 1.0)

    return model
end

# this fn returns a video simulation of a wolf pack hunting a single reactive/escaping prey
function freactive_hunt_sim(total_wolf = 6, total_sheep = 25, circ_dist = 1.0, d_rep = 1.0, w_rep_ww = 0.01,
                             a = 1, b = 0.8, c = 0.8,
                             wolf_speed = 2.0, sheep_speed = 1.0, wolf_mass = 1.0, sheep_mass = 1.0, 
                             n_att = 5, n_ali = 2,
                             w_rep_ws = 2, w_rep_ss = 2, w_att = 1.0, w_ali = 0.8,
                             dt = 0.25, size = (40.0, 40.0), seed = 124, framerate = 5, frames = 40)

    model = initialize(; size, total_wolf, total_sheep,
                       circ_dist, d_rep,
                       w_rep_ww,
                       a, b, c,
                       wolf_speed,
                       sheep_speed,
                       wolf_mass,
                       sheep_mass,
                       n_att, n_ali,
                       w_rep_ws,
                       w_rep_ss,
                       w_att,
                       w_ali,
                       dt, seed)

    # Define color, size, and marker functions
    agent_color(a::Sheep) = :green
    agent_color(a::Wolf) = :blue
    agent_size(a::Sheep) = 10
    agent_size(a::Wolf) = 8
    agent_marker(a::Sheep) = 'o'
    agent_marker(a::Wolf) = :diamond

    # Create the animation
    abmvideo("wolf_hunt.mp4", model;
    title = "Reactive Sheep Hunt Simulation", framerate, frames, agent_color, agent_size, agent_marker)

end

freactive_hunt_sim(
    6,                      # total_wolf
    25,                     # total_sheep
    5.0,                    # circ_dist
    1.0,                    # d_rep
    0.25,                    # w_rep_ww
    1.0,                    # a
    0.5,                    # b
    0.8,                    # c
    2.0,                    # wolf_speed
    2.0,                    # sheep_speed
    1.0,                    # wolf_mass
    1.0,                    # sheep_mass
    5,                      # n_att
    3,                      # n_ali
    5.0,                    # w_rep_ws
    10.0,                    # w_rep_ss
    5.0,                    # w_att
    3.0,                    # w_ali
    0.05,                   # dt
    (80.0, 80.0),           # size of sim space
    100,                    # seed 
    100,                      # framerate
    1000                      # frames
)

println("success!")