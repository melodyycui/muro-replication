using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro
using StatsBase: sample

# update variables/constants list
# figure out how to push 

# constructing wolf agent type:
@agent struct Wolf(ContinuousAgent{2, Float64})

    target_id::Union{Nothing, Int} # ID of the current target sheep
    time_on_target::Float64 # time spent chasing the current target sheep
    was_circling::Bool # whether the wolf was circling a sheep in the last step

end

# constructing sheep agent type:
@agent struct Sheep(ContinuousAgent{2, Float64})
end


# this fn updates the position and velocity of a wolf agent after one time step, dt:
function animal_step!(agent::Wolf, model)

    mass_wolf = 1

    # Wolf-Wolf Repulsion: Wolves want to avoid each other:
    ww_repulsionulsion = [0.0, 0.0]

    for neighbor in allagents(model)
        if neighbor.id != agent.id && isa(neighbor, Wolf)

        dist = norm(agent.pos - neighbor.pos)

            # The following contains a 'Divide by Zero' error prevention (assuming that it is very 
            # unlikely that two wolves will spawn in the same location). Can replace with 
            # ww_repulsionulsion += model.ww_force_coefficient * (agent.pos - neighbor.pos) / (dist^2 + ε) 
            # where ε is a small positive number like 1e-8, if the above assumption is not valid.
            if dist > 1e-8
                
                ww_repulsionulsion += model.ww_force_coefficient * (agent.pos - neighbor.pos) / dist^2

            end

        end

    end

    # Obtain a list of all sheep agents in the model
    sheep_list = [s for s in allagents(model) if isa(s, Sheep)]

    # Sort all sheep by distance away from the wolf agent (closest first)
    sorted_sheep = sort(sheep_list, by = s -> norm(agent.pos - s.pos))

    # If there are no sheep, stop
    if isempty(sorted_sheep)

        agent.vel = [0.0, 0.0]
        return

    end

    # Initialize target sheep if none exists
    if isnothing(agent.target_id)

        agent.target_id = sorted_sheep[1].id
        agent.time_on_target = 0.0 # Set the time our wolf agent has spent chasing the current target to 0

    end

    # Find current target sheep by ID
    index = findfirst(s -> s.id == agent.target_id, sorted_sheep)
    target_sheep = isnothing(index) ? nothing : sorted_sheep[index]

    # If target sheep disappeared, reset to new closest
    if target_sheep === nothing

        agent.target_id = sorted_sheep[1].id # Reset target ID to the closest sheep
        agent.time_on_target = 0.0 # Reset time on target to 0
        target_sheep = sorted_sheep[1]

    end

    # Check target sheep speed
    target_sheep_speed = norm(target_sheep.vel)

    # Assumption: If wolves are 'successfully' chasing down a target sheep, the target must be slowing down. 
    # After a certain length of time, agent.time_on_target, during a 'successful' chase, 
    #the target sheep's speed should be below a threshold, sheep_threshold. 
    # Otherwise, the wolf agent will switch to a new target sheep.

    sheep_speed_threshold = 0 

    # Increment time spent by current wolf agent chasing down the current target sheet
    agent.time_on_target += model.dt

    # Switch target sheep if target is still faster than the sheep_speed_threshold after 7 seconds
    # NOTE: 7 seconds is a placeholder value, I will make it a parameter once feedback has been given to this model
    if agent.time_on_target >= 7.0 && target_sheep_speed > sheep_speed_threshold

        for s in sorted_sheep

            # The sorted_sheep list is sorted by distance, so we can break as soon as 
            # we find a sheep that is not the current target:
            if s.id != agent.target_id

                agent.target_id = s.id
                agent.time_on_target = 0.0
                target_sheep = s
                break

            end

        end

    end

    # Compute distance from current wolf agent to target sheep
    current_wolf_sheep_distance = norm(agent.pos - target_sheep.pos)

    # Circling logic
    circling = false

    # If the wolf is close enough, some wolf_orbit_dist_threshold, to the sheep, it will circle around it
    if current_wolf_sheep_distance <= model.wolf_orbit_dist_threshold && current_wolf_sheep_distance > 1e-8

        to_sheep = agent.pos - target_sheep.pos

        rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)] # 90-degree rotation matrix

        # Compute the tangential direction for wolf's circling behaviour
        u = rotation_matrix * to_sheep

        dot_product = dot(u, ww_repulsionulsion)
        denom = norm(u)^2

        if denom > 1e-8

            # Project the wolf-wolf repulsion onto the perpendicular (circling) direction
            # Then scale to the desired wolf chase speed to determine final velocity
            proj_u_v = (dot_product / denom) * u * model.wolf_chase_speed
            agent.vel = proj_u_v
            circling = true

        end

        # If this wolf wasn't circling target sheep in last step, increment count of wolves circling the sheep
        if !agent.was_circling

            model.sheep_trapped_by[target_sheep.id] = get(model.sheep_trapped_by, target_sheep.id, 0) + 1

        end

        agent.was_circling = true

    end

    # If the wolf is not close enough to circle, it will chase the target sheep
    if !circling && current_wolf_sheep_distance > 1e-8

        # If the wolf was circling the sheep in the last step, decrement count of wolves circling the sheep
        if agent.was_circling

            model.sheep_trapped_by[target_sheep.id] = max(0, get(model.sheep_trapped_by, target_sheep.id, 0) - 1)
            
        end

        agent.was_circling = false

        # Calculate the direction vector from the wolf to the target sheep
        unit_vec = (target_sheep.pos - agent.pos) / current_wolf_sheep_distance

        # Model wolf-sheep attraction as a decaying exponential function of distance
        a, b = 50, 0.05
        ws_attraction = a * exp(-b * current_wolf_sheep_distance) * unit_vec

        # Net force on wolf agent is sum of wolf-wolf repulsion and wolf-sheep attraction forces acting on it:
        net_force_on_wolf = (ww_repulsionulsion + ws_attraction)  

        # update the velocity of the wolf agent:
        agent.vel += (net_force_on_wolf/ mass_wolf) * model.dt

    end

    # Capping the speed of the wolf agent to a max_wolf_speed
    max_wolf_speed = 10.0  # can adjust this value as needed
    current_wolf_speed = norm(agent.vel)

    if current_wolf_speed > max_wolf_speed

        agent.vel = (agent.vel / current_wolf_speed) * max_wolf_speed

    end

    move_agent!(agent, model, model.dt)

end

# this fn updates the position and velocity of a sheep agent after one time step, dt:
function animal_step!(agent::Sheep, model)

    # Model assumption: Once at least 3 wolves are circling a sheep, the sheep is considered 'sufficiently trapped'
    # and will stop moving. This is to prevent the sheep from moving indefinitely and allow wolves to actually 
    # encircle and 'capture' the sheep.

    # Check if this sheep is 'sufficiently trapped'
    num_wolves_sufficiently_trapping = get(model.sheep_trapped_by, agent.id, 0)

    if num_wolves_sufficiently_trapping >= 3

        agent.vel = [0.0, 0.0] # sheep is trapped, so stop moving
        return

    end

    mass_sheep = 1

    # Wolf-Sheep Repulsion: sheep agent wants to move away from wolf agents
    ws_repulsion = [0, 0] # stores net repulsive force from wolves on sheep
  
    # each wolf exerts a repulsive force on the sheep agent
    for wolf in allagents(model)

        if isa(wolf, Wolf)

           # placeholder values
           a = 5.0
           b = 0.1
          
           # sum up the repulsive forces exerted by each wolf on the sheep
            dist_btwn_wolf_n_sheep = norm(agent.pos - wolf.pos)

            if dist_btwn_wolf_n_sheep > 1e-8

                direction_of_wolf_sheep_rep = (agent.pos - wolf.pos) / dist_btwn_wolf_n_sheep
                ws_repulsion += (a * exp(-b * dist_btwn_wolf_n_sheep)) * direction_of_wolf_sheep_rep

            end

        end

    end


    # initialize num of sheep that exert attraction, alignment, and repulsion forces on current sheep agent:
    num_of_attractive_sheep = 3
    num_of_alignment_sheep = 2
    num_of_repulsive_sheep = 0


    # initialize net sheep-sheep attraction, repulsion, and alignment forces:
    ss_attraction = [0, 0]
    ss_repulsion = [0, 0]
    ss_alignment = [0, 0]


    # distance threshold for sheep-sheep repulsion
    dist_threshold_ss_repulsion = 1.0      # need to replace
  
    # weights for the attraction, alignment, and repulsion forces:
    w_attraction = 0.3
    w_repulsion = 0.6
    w_alignment = 0.2

   # sheep that are too close to our current sheep agent exert a repulsive force on the agent:
    for sheep in allagents(model)

        if isa(sheep, Sheep) && sheep.id != agent.id

            to_sheep = sheep.pos - agent.pos
            dist_btwn_sheep = norm(to_sheep)

            if dist_btwn_sheep > 1e-8 && dist_btwn_sheep < dist_threshold_ss_repulsion

                ss_repulsion -= (w_repulsion * to_sheep / dist_btwn_sheep)
                num_of_repulsive_sheep += 1

            end

        end

    end


    # the repulsive force felt by current sheep agent is average of the repulsive sheep-sheep forces
    # exerted on it by all nearby sheep:
    if (num_of_repulsive_sheep > 0)

       ss_repulsion /= num_of_repulsive_sheep

    end

    # Find all nearby sheep agents that are not the current sheep agent:
    neighbors = [i for i in nearby_agents(agent, model) if isa(i, Sheep) && i.id != agent.id]

    if isempty(neighbors)

        return  # no nearby agents — skip all social behavior

    end

    # Find random sample (w/0 replacement) of num_of_attractive_sheep sheep neighbors for the current sheep to be attracted to:
    sample_of_attractive_neighbors = sample(neighbors, min(num_of_attractive_sheep, length(neighbors)); replace=false)

    for neighbor in sample_of_attractive_neighbors

        to_sheep = neighbor.pos - agent.pos
        dist_btwn_sheep = norm(to_sheep)

        if dist_btwn_sheep > 1e-8

            ss_attraction += (w_attraction / num_of_attractive_sheep) * (to_sheep / dist_btwn_sheep)

        end

    end

    # Find random sample (w/0 replacement) of num_of_alignment_sheep sheep neighbors for the current sheep to align with:
    sample_of_alignment_neighbors = sample(neighbors, min(num_of_alignment_sheep, length(neighbors)); replace=false)

    for neighbor in sample_of_alignment_neighbors

        dist_btwn_sheep = norm(neighbor.pos)

        if dist_btwn_sheep > 1e-8

            ss_alignment += (w_alignment / num_of_alignment_sheep) * (neighbor.pos / dist_btwn_sheep)

        end

    end

    # net force on the sheep agent is sum of the wolf-sheep repulsion, sheep-sheep repulsion,
    # sheep-sheep attraction, and sheep-sheep alignment forces acting on it:
    net_force_on_sheep = ws_repulsion + ss_repulsion + ss_attraction + ss_alignment

    # update the velocity of the sheep agent:
    agent.vel += (net_force_on_sheep/mass_sheep)*model.dt

    # Capping the speed of the sheep agent to a max_sheep_speed
    max_sheep_speed = 5.0  # can adjust this value as needed
    current_sheep_speed = norm(agent.vel)

    if current_sheep_speed > max_sheep_speed

        agent.vel = (agent.vel / current_sheep_speed) * max_sheep_speed

    end

    move_agent!(agent, model, model.dt)

end


# this fn initializes our model with wolves randomly generated in the upper left quadrant, [5, 90] to [15, 95],
# and sheep randomly generated in the center of the simulation space, [45, 45] to [65, 65]:
function initialize(; size, wolf_orbit_dist_threshold, ww_force_coefficient,
    sw_force_coefficient, wolf_chase_speed,
    dt, seed, center, sheep_speed, num_wolf, num_sheep)


    space = ContinuousSpace(size; periodic = false)


    properties = Dict(
        :wolf_orbit_dist_threshold => wolf_orbit_dist_threshold,
        :ww_force_coefficient => ww_force_coefficient,
        :sw_force_coefficient => sw_force_coefficient,
        :wolf_chase_speed => wolf_chase_speed,
        :dt => dt,
        :center => center,
        :sheep_speed => sheep_speed,
        :sheep_trapped_by => Dict{Int, Int}()
    )


    rng = Xoshiro(seed)


    model = StandardABM(Union{Wolf, Sheep}, space;
        properties=properties,
        agent_step! = animal_step!,
        rng=rng,
        container=Vector,
        scheduler=Schedulers.Randomly())


   for _ in 1:num_wolf

        rand_pos = [5 + 10 * rand(rng), 90 + 5 * rand(rng)]  # x: 5–15, y: 90–95
        add_agent!(Wolf, model; pos = rand_pos, vel = (0.0, 0.0), target_id = nothing, 
        time_on_target = 0.0, was_circling = false)

   end


    for _ in 1:num_sheep

        rand_pos = [45 + 20 * rand(rng), 45 + 20 * rand(rng)]  # x: 45–65, y: 45–65
        add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0))
        
    end

    return model

end


# this fn make simulation video of the wolves hunting sheep:
function multi_hunt_sim(; wolf_orbit_dist_threshold=0.5, ww_force_coefficient=1.0,
    sw_force_coefficient=2.0, wolf_chase_speed=1.0, dt=0.1,
    size=(20.0, 20.0), seed=125, framerate=15, frames=200,
    center=[10.0, 10.0], sheep_speed=0.5, num_wolf=3, num_sheep=5)


    model = initialize(
        size=size,
        wolf_orbit_dist_threshold=wolf_orbit_dist_threshold,
        ww_force_coefficient=ww_force_coefficient,
        sw_force_coefficient=sw_force_coefficient,
        wolf_chase_speed=wolf_chase_speed,
        dt=dt,
        seed=seed,
        center=center,
        sheep_speed=sheep_speed,
        num_wolf=num_wolf,
        num_sheep=num_sheep
    )


    ac(a::Wolf) = :green
    ac(a::Sheep) = :blue
    as(a::Wolf) = 10
    as(a::Sheep) = 13
    am(a::Wolf) = 'o'
    am(a::Sheep) = :diamond


    abmvideo("wolf_hunt.mp4", model;
        title="Multibb Wolf–Sheep Physics Hunt",
        framerate=framerate,
        frames=frames,
        ac=ac, as=as, am=am)

end

# testing
multi_hunt_sim(
    wolf_orbit_dist_threshold=5.0,      # distance at which wolves will start circling sheep
    ww_force_coefficient=1.0,   # wolf-wolf repulsion coefficient
    sw_force_coefficient=5.0,   # wolf-sheep attraction coefficient
    wolf_chase_speed=1.0,       # speed at which wolves chase sheep
    dt=0.05,                    # time step for the simulation
    size=(100.0, 100.0),        # size of the simulation space
    seed=124,                   # random seed for reproducibility
    framerate=15,               # frames per second for the video
    frames=2000,                # number of frames in the video
    center=[50.0, 50.0],        # center of the simulation space
    sheep_speed=0.75,           # speed of the sheep
    num_wolf=6,                 # number of wolves in the simulation
    num_sheep=30                # number of sheep in the simulation
) 