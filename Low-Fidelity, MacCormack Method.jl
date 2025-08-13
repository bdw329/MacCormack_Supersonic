using Plots
using LinearAlgebra
using Interpolations
using NearestNeighbors
using GeometryBasics
using Statistics
using Colors, ColorSchemes

mutable struct SimulationParams
    ρ::Array{Float64,2}
    p::Array{Float64,2}
    u::Array{Float64,2}
    v::Array{Float64,2}
    T::Array{Float64,2}
    P::Array{Float64,2}  # log pressure
    R::Array{Float64,2}  # log density
    ψ::Array{Float64,2}  # log pressure minus γ log density
end

mutable struct GridVars
    x_grid::Array{Float64,2}
    y_grid::Array{Float64,2}
    xb::Array{Float64,1}
    xs::Array{Float64,1}
    Z::Array{Float64,2}
    W::Array{Float64,1}
    W_new::Array{Float64,1}
end

mutable struct inputs
    γ::Float64
    R_gas::Float64
    CFL::Float64
    Nx::Int
    Ny::Int
    M_inf::Float64
    T_inf::Float64
    p_inf::Float64
    ρ_inf::Float64
    R_body::Float64
end

mutable struct derivatives
    dudt::Array{Float64,2}
    dvdt::Array{Float64,2}
    dRdt::Array{Float64,2}
    dψdt::Array{Float64,2}
    dpdt::Array{Float64,2}
    dρdt::Array{Float64,2}
    dudt_::Array{Float64,2}
    dvdt_::Array{Float64,2}
    dRdt_::Array{Float64,2}
    dψdt_::Array{Float64,2}
    dpdt_::Array{Float64,2}
    dρdt_::Array{Float64,2}
    dudt_avg::Array{Float64,2}
    dvdt_avg::Array{Float64,2}
    dRdt_avg::Array{Float64,2}
    dψdt_avg::Array{Float64,2}
    dpdt_avg::Array{Float64,2}
    dρdt_avg::Array{Float64,2}
end

mutable struct corrector
    u_::Array{Float64,2}
    v_::Array{Float64,2}
    R_::Array{Float64,2}
    ψ_::Array{Float64,2}
    p_::Array{Float64,2}
    ρ_::Array{Float64,2}
    P_::Array{Float64,2}  
    T_::Array{Float64,2}
end

function initialize_params(Nx, Ny, ρ_inf, p_inf, M_inf, T_inf, γ, R_gas)
    ρ = fill(ρ_inf, Ny, Nx)
    p = fill(p_inf, Ny, Nx)
    u = fill(M_inf*sqrt(γ*T_inf*R_gas), Ny, Nx)
    v = fill(0, Ny, Nx)
    T = fill(T_inf, Ny, Nx)

    P = fill(log(p_inf), Ny, Nx)  # log pressure
    R = fill(log(ρ_inf), Ny, Nx)  # log density
    ψ = fill(log(p_inf) - γ * log(ρ_inf), Ny, Nx)
    return SimulationParams(ρ, p, u, v, T, P, R, ψ)
end

function initialize_GridVars(Nx, Ny)
    x_grid = zeros(Ny, Nx)
    y_grid = zeros(Ny, Nx)
    xb = zeros(Ny)
    xs = zeros(Ny)
    Z = zeros(Ny, Nx)
    W = zeros(Ny)
    W_new = zeros(Ny)
    return GridVars(x_grid, y_grid, xb, xs, Z, W, W_new)
end

function initialize_inputs(γ, R_gas, CFL, Nx, Ny, M_inf, T_inf, p_inf, ρ_inf, R_body)
    return inputs(γ, R_gas, CFL, Nx, Ny, M_inf, T_inf, p_inf, ρ_inf, R_body)
end

function initialize_derivatives(Nx, Ny)
    dudt = zeros(Ny, Nx)
    dvdt = zeros(Ny, Nx)
    dRdt = zeros(Ny, Nx)
    dψdt = zeros(Ny, Nx)
    dpdt = zeros(Ny, Nx)
    dρdt = zeros(Ny, Nx)
    dudt_ = zeros(Ny, Nx)
    dvdt_ = zeros(Ny, Nx)
    dRdt_ = zeros(Ny, Nx)
    dψdt_ = zeros(Ny, Nx)
    dpdt_ = zeros(Ny, Nx)
    dρdt_ = zeros(Ny, Nx)
    dudt_avg = zeros(Ny, Nx)
    dvdt_avg = zeros(Ny, Nx)
    dRdt_avg = zeros(Ny, Nx)
    dψdt_avg = zeros(Ny, Nx)
    dpdt_avg = zeros(Ny, Nx)
    dρdt_avg = zeros(Ny, Nx)
    return derivatives(dudt, dvdt, dRdt, dψdt, dpdt, dρdt,
                       dudt_, dvdt_, dRdt_, dψdt_, dpdt_, dρdt_,
                       dudt_avg, dvdt_avg, dRdt_avg, dψdt_avg, dpdt_avg, dρdt_avg)
end

function initialize_corrector(Nx, Ny)
    u_ = zeros(Ny, Nx)
    v_ = zeros(Ny, Nx)
    R_ = zeros(Ny, Nx)
    ψ_ = zeros(Ny, Nx)
    p_ = zeros(Ny, Nx)
    ρ_ = zeros(Ny, Nx)
    P_ = zeros(Ny, Nx)
    T_ = zeros(Ny, Nx)
    return corrector(u_, v_, R_, ψ_, p_, ρ_, P_, T_)
end

function form_domain(params, grid, inputs)
    # === YOUR ORIGINAL SHOCK SHAPE CODE (UNCHANGED) ===
    y_body = LinRange(0, inputs.R_body, 300)
    x_body = -sqrt.(1 .- y_body.^2)

    A_term = 0.5
    B_term = 0.5
    delta = inputs.R_body * (A_term / (inputs.M_inf^2 - 1) + B_term / sqrt(inputs.γ + 1))
    a_shock = inputs.R_body * inputs.M_inf * (inputs.M_inf - sqrt(inputs.M_inf))
    b_shock = inputs.R_body * sqrt(inputs.M_inf * (inputs.M_inf - sqrt(inputs.M_inf)))
    x0 = -inputs.R_body - delta - a_shock

    y_grid_1d = LinRange(0, inputs.R_body, inputs.Ny)
    grid.xb = -sqrt.(1 .- y_grid_1d.^2)
    grid.xs = x0 .+ sqrt.(a_shock^2 .* (1 .+ y_grid_1d.^2 ./ b_shock^2))

    grid.x_grid = zeros(inputs.Ny, inputs.Nx)
    grid.y_grid = zeros(inputs.Ny, inputs.Nx)
    for i in 1:inputs.Ny
        grid.x_grid[i, :] .= range(grid.xb[i], grid.xs[i], length = inputs.Nx)
        grid.y_grid[i, :] .= y_grid_1d[i]
    end
    grid.Z = zeros(inputs.Ny, inputs.Nx)
    for i in 1:inputs.Ny
        for j in 1:inputs.Nx
            x = grid.x_grid[i, j]
            grid.Z[i, j] = (x - grid.xb[i]) / (grid.xs[i] - grid.xb[i])
        end
    end
    grid.W = zeros(inputs.Ny)

    # === FLOW INITIALIZATION ADDITIONS ===
    W_inf = inputs.M_inf * sqrt(inputs.γ * inputs.R_gas * inputs.T_inf)
    y_profile = collect(y_grid_1d)

    # --- Body wall (column 1) slip condition ---
    for i in 1:inputs.Ny
        params.u[i,1] = 0.0
        params.v[i,1] = 0.0
        # assume stagnation pressure/density near wall initially (simple approach)
        params.p[i,1] = max(inputs.p_inf * (1 + 0.5*(inputs.γ-1)*inputs.M_inf^2)^(inputs.γ/(inputs.γ-1)), 1e-12)
        params.ρ[i,1] = max(inputs.ρ_inf * (params.p[i,1]/inputs.p_inf)^(1/inputs.γ), 1e-12)
        params.T[i,1] = params.p[i,1] / (params.ρ[i,1] * inputs.R_gas)
        params.R[i,1] = log(params.ρ[i,1])
        params.ψ[i,1] = log(params.p[i,1]) - inputs.γ*params.R[i,1]
        params.P[i,1] = log(params.p[i,1])
    end

    # --- Shock boundary (column Nx) from oblique shock relations ---
    for i in 1:inputs.Ny
        β = β_func(i, grid.xs, y_profile)
        M1n = inputs.M_inf * sin(β)

        p2_p1 = 1 + 2*inputs.γ/(inputs.γ+1)*(M1n^2 - 1)
        ρ2_ρ1 = ((inputs.γ+1)*M1n^2) / ((inputs.γ-1)*M1n^2 + 2)
        T2_T1 = p2_p1 / ρ2_ρ1

        params.p[i,inputs.Nx] = max(inputs.p_inf * p2_p1, 1e-12)
        params.ρ[i,inputs.Nx] = max(inputs.ρ_inf * ρ2_ρ1, 1e-12)
        params.T[i,inputs.Nx] = max(inputs.T_inf * T2_T1, 1e-12)

        # Velocity components
        Vn2 = W_inf * sin(β) / ρ2_ρ1
        Vt1 = W_inf * cos(β)
        Vt2 = Vt1
        params.u[i,inputs.Nx] = Vn2 * sin(β) + Vt2 * cos(β)
        params.v[i,inputs.Nx] = -Vn2 * cos(β) + Vt2 * sin(β)

        params.R[i,inputs.Nx] = log(params.ρ[i,inputs.Nx])
        params.ψ[i,inputs.Nx] = log(params.p[i,inputs.Nx]) - inputs.γ*params.R[i,inputs.Nx]
        params.P[i,inputs.Nx] = log(params.p[i,inputs.Nx])
    end

    # --- Interior interpolation (now from wall to shock) ---
    for i in 1:inputs.Ny
        for j in 2:(inputs.Nx-1)
            λ = (j-1)/(inputs.Nx-1)
            params.u[i,j] = (1-λ)*params.u[i,1] + λ*params.u[i,inputs.Nx]
            params.v[i,j] = (1-λ)*params.v[i,1] + λ*params.v[i,inputs.Nx]
            params.p[i,j] = max((1-λ)*params.p[i,1] + λ*params.p[i,inputs.Nx], 1e-12)
            params.ρ[i,j] = max((1-λ)*params.ρ[i,1] + λ*params.ρ[i,inputs.Nx], 1e-12)
            params.T[i,j] = max((1-λ)*params.T[i,1] + λ*params.T[i,inputs.Nx], 1e-12)

            params.R[i,j] = log(params.ρ[i,j])
            params.ψ[i,j] = log(params.p[i,j]) - inputs.γ*params.R[i,j]
            params.P[i,j] = log(params.p[i,j])
        end
    end

    return grid.x_grid, grid.y_grid, grid.xb, grid.xs, grid.Z, grid.W
end

# Step 4: Transform grid into rectangular spaced grid ############################
function transform_grid(params, grid, inputs)
    for i in 1:inputs.Ny
        for j in 1:inputs.Nx
            x = grid.x_grid[i, j]            # <<< fixed i–j order
            grid.Z[i, j]  = (x - grid.xb[i])/(grid.xs[i] - grid.xb[i])
        end
    end
    return Z = grid.Z
end

# Step 5: Transform dependent variables ###########################################
function P_func(p)
    return log(max(p, 1e-12))
end

function R_func(ρ)
    return log(max(ρ, 1e-12))
end

function ψ_func(p, ρ, inputs)
    return P_func(p) - inputs.γ*R_func(ρ)
end

function C_func(Z, i, j, Ny, y_grid, xs, xb)
    # Calculate dbdy using index i (rows = y-direction)
    if i == 1
        dbdy = (xb[2] - xb[1]) / (y_grid[2] - y_grid[1])
    elseif i == Ny
        dbdy = (xb[Ny] - xb[Ny-1]) / (y_grid[Ny] - y_grid[Ny-1])
    elseif abs(y_grid[i+1] - y_grid[i-1]) < 1e-12  # protect against division by zero
        dbdy = 0.0
    else
        dbdy = (xb[i+1] - xb[i-1]) / (y_grid[i+1] - y_grid[i-1])
    end

    # Calculate theta using index i
    if i == 1
        theta = atan(y_grid[2] - y_grid[1], xs[2] - xs[1])
    elseif i == Ny
        theta = atan(y_grid[Ny] - y_grid[Ny-1], xs[Ny] - xs[Ny-1])
    else
        theta = atan(y_grid[i+1] - y_grid[i-1], xs[i+1] - xs[i-1])
    end

    ε = 1e-4   # minimum allowed slope

    tanθ = tan(theta)
    cotθ = abs(tanθ) < ε ? 1/ε : 1 / tanθ

    return (Z - 1) * dbdy - Z * cotθ
end

function B_func(u, W, Z, v, C, i, j, delta) 
    return (u - W * Z + v * C) / delta
end

function β_func(i, xs::Vector{Float64}, y_profile::Vector{Float64})
    # central difference slope of the shock curve
    if i == 1
        dy = y_profile[2] - y_profile[1]
        dx = xs[2] - xs[1]
    elseif i == length(xs)
        dy = y_profile[end] - y_profile[end-1]
        dx = xs[end] - xs[end-1]
    else
        dy = y_profile[i+1] - y_profile[i-1]
        dx = xs[i+1] - xs[i-1]
    end

    # β is angle between shock and freestream (x-axis)
    β = atan(abs(dy/dx))      # positive angle
    return β                  # no artificial clamping
end

# Time-stepping parameters ##########################################################################################
function update_dt(params, grid, inputs)
    dt_min = typemax(Float64)

    for i in 1:inputs.Ny
        for j in 1:inputs.Nx
            # local grid spacing in x and y
            dx = (j < inputs.Nx)  ? 
                 (grid.x_grid[i, j+1] - grid.x_grid[i, j]) :
                 (grid.x_grid[i, j]   - grid.x_grid[i, j-1])

            dy = (i < inputs.Ny)  ? 
                 (grid.y_grid[i+1, j] - grid.y_grid[i, j]) :
                 (grid.y_grid[i, j]   - grid.y_grid[i-1, j])

            a_local = sqrt(inputs.γ * inputs.R_gas * params.T[i,j])
            dt_local = inputs.CFL * min(
                abs(dx) / (abs(params.u[i,j]) + a_local),
                abs(dy) / (abs(params.v[i,j]) + a_local)
            )
            dt_min = min(dt_min, dt_local)
        end
    end
    return dt_min
end

# iterate points in domain ########################################################################################################
function interior_points(params, grid, inputs, derivatives, corrector, dt)
    # Uniform grid spacing (still an approximation)
    dZ = 1/(inputs.Nx - 1)
    dy = 1/(inputs.Ny - 1)

    # --------------------
    # Predictor Pass
    # --------------------
    for i in 2:(inputs.Ny - 1)
        for j in 2:(inputs.Nx - 1)
            delta = grid.xs[i] - grid.xb[i]
            C = C_func(grid.Z[i,j], i, j, inputs.Ny, grid.y_grid, grid.xs, grid.xb)
            B = B_func(params.u[i,j], grid.W[i], grid.Z[i,j], params.v[i,j], C, i, j, delta)

            # Spatial derivatives (forward differences)
            dudZ = (params.u[i,j+1] - params.u[i,j]) / dZ
            dudy = (params.u[i+1,j] - params.u[i,j]) / dy
            dvdZ = (params.v[i,j+1] - params.v[i,j]) / dZ
            dvdy = (params.v[i+1,j] - params.v[i,j]) / dy
            dPdZ = (params.P[i,j+1] - params.P[i,j]) / dZ
            dPdy = (params.P[i+1,j] - params.P[i,j]) / dy
            dRdZ = (params.R[i,j+1] - params.R[i,j]) / dZ
            dRdy = (params.R[i+1,j] - params.R[i,j]) / dy
            dψdZ = (params.ψ[i,j+1] - params.ψ[i,j]) / dZ
            dψdy = (params.ψ[i+1,j] - params.ψ[i,j]) / dy

            # Use log-pressure derivative for dp/dZ
            dpdZ = params.p[i,j] * dPdZ

            # Predictor time derivatives
            derivatives.dudt[i,j] = -(B*dudZ + params.v[i,j]*dudy +
                                      (params.p[i,j]/(params.ρ[i,j]*delta))*dpdZ)
            derivatives.dvdt[i,j] = -(B*dvdZ + params.v[i,j]*dvdy +
                                      (params.p[i,j]*C)/(params.ρ[i,j]*delta)*dPdZ +
                                      (params.p[i,j]/params.ρ[i,j])*dPdy)
            derivatives.dRdt[i,j] = -(B*dRdZ + (1/delta)*dudZ +
                                      (C/delta)*dvdZ + dvdy + params.v[i,j]*dRdy)
            derivatives.dψdt[i,j] = -(B*dψdZ + params.v[i,j]*dψdy)

            # Predictor variable updates
            corrector.u_[i,j] = params.u[i,j] + derivatives.dudt[i,j]*dt
            corrector.v_[i,j] = params.v[i,j] + derivatives.dvdt[i,j]*dt
            corrector.R_[i,j] = params.R[i,j] + derivatives.dRdt[i,j]*dt
            corrector.ψ_[i,j] = params.ψ[i,j] + derivatives.dψdt[i,j]*dt

            # Compute physical variables from transformed ones
            corrector.ρ_[i,j] = exp(corrector.R_[i,j])
            corrector.p_[i,j] = exp(corrector.ψ_[i,j] + inputs.γ * corrector.R_[i,j])
            corrector.P_[i,j] = log(max(corrector.p_[i,j], 1e-12))
            corrector.T_[i,j] = corrector.p_[i,j] / (corrector.ρ_[i,j] * inputs.R_gas)
        end
    end

    # --------------------
    # Corrector Pass
    # --------------------
    for i in 2:(inputs.Ny - 1)
        for j in 2:(inputs.Nx - 1)
            delta = grid.xs[i] - grid.xb[i]
            C_ = C_func(grid.Z[i,j], i, j, inputs.Ny, grid.y_grid, grid.xs, grid.xb)
            B_ = B_func(corrector.u_[i,j], grid.W[i], grid.Z[i,j],
                        corrector.v_[i,j], C_, i, j, delta)

            # Spatial derivatives (rearward differences)
            dudZ_ = (corrector.u_[i,j] - corrector.u_[i,j-1]) / dZ
            dudy_ = (corrector.u_[i,j] - corrector.u_[i-1,j]) / dy
            dvdZ_ = (corrector.v_[i,j] - corrector.v_[i,j-1]) / dZ
            dvdy_ = (corrector.v_[i,j] - corrector.v_[i-1,j]) / dy
            dPdZ_ = (corrector.P_[i,j] - corrector.P_[i,j-1]) / dZ
            dPdy_ = (corrector.P_[i,j] - corrector.P_[i-1,j]) / dy
            dRdZ_ = (corrector.R_[i,j] - corrector.R_[i,j-1]) / dZ
            dRdy_ = (corrector.R_[i,j] - corrector.R_[i-1,j]) / dy
            dψdZ_ = (corrector.ψ_[i,j] - corrector.ψ_[i,j-1]) / dZ
            dψdy_ = (corrector.ψ_[i,j] - corrector.ψ_[i-1,j]) / dy

            # Use log-pressure derivative for dp/dZ
            dpdZ_ = corrector.p_[i,j] * dPdZ_

            # Corrector time derivatives
            derivatives.dudt_[i,j] = -(B_*dudZ_ + corrector.v_[i,j]*dudy_ +
                                       (corrector.p_[i,j]/(corrector.ρ_[i,j]*delta))*dpdZ_)
            derivatives.dvdt_[i,j] = -(B_*dvdZ_ + corrector.v_[i,j]*dvdy_ +
                                       (corrector.p_[i,j]*C_)/(corrector.ρ_[i,j]*delta)*dPdZ_ +
                                       (corrector.p_[i,j]/corrector.ρ_[i,j])*dPdy_)
            derivatives.dRdt_[i,j] = -(B_*dRdZ_ + (1/delta)*dudZ_ +
                                       (C_/delta)*dvdZ_ + dvdy_ + corrector.v_[i,j]*dRdy_)
            derivatives.dψdt_[i,j] = -(B_*dψdZ_ + corrector.v_[i,j]*dψdy_)

            # Average derivatives
            derivatives.dudt_avg[i,j] = 0.5*(derivatives.dudt[i,j] + derivatives.dudt_[i,j])
            derivatives.dvdt_avg[i,j] = 0.5*(derivatives.dvdt[i,j] + derivatives.dvdt_[i,j])
            derivatives.dRdt_avg[i,j] = 0.5*(derivatives.dRdt[i,j] + derivatives.dRdt_[i,j])
            derivatives.dψdt_avg[i,j] = 0.5*(derivatives.dψdt[i,j] + derivatives.dψdt_[i,j])

            # Final update (n+1)
            params.u[i,j] += derivatives.dudt_avg[i,j]*dt
            params.v[i,j] += derivatives.dvdt_avg[i,j]*dt
            params.R[i,j] += derivatives.dRdt_avg[i,j]*dt
            params.ψ[i,j] += derivatives.dψdt_avg[i,j]*dt

            # Recompute physical variables from transformed ones
            params.ρ[i,j] = exp(params.R[i,j])
            params.p[i,j] = exp(params.ψ[i,j] + inputs.γ * params.R[i,j])
            params.P[i,j] = log(max(params.p[i,j], 1e-12))
            params.T[i,j] = params.p[i,j] / (params.ρ[i,j] * inputs.R_gas)
        end
    end
    return params
end

function shock_boundary(params, grid, inputs, dt)
    j = inputs.Nx
    W_inf = inputs.M_inf * sqrt(inputs.γ * inputs.R_gas * inputs.T_inf)
    y_profile = grid.y_grid[:, 1]

    # --- Step 1: Apply shock relations to boundary ---
    for i in 2:(inputs.Ny - 1)
        # Shock angle: normal near centerline
        β = β_func(i, grid.xs, y_profile)
        if abs(y_profile[i]) < 0.05
            β = π/2
        end

        # Upstream normal Mach
        M1n = inputs.M_inf * sin(β)

        # Shock jump conditions
        p2_p1 = 1 + 2 * inputs.γ / (inputs.γ + 1) * (M1n^2 - 1)
        ρ2_ρ1 = ((inputs.γ + 1) * M1n^2) / ((inputs.γ - 1) * M1n^2 + 2)
        T2_T1 = p2_p1 / ρ2_ρ1

        # Thermodynamic state
        params.p[i, j] = inputs.p_inf * p2_p1
        params.ρ[i, j] = inputs.ρ_inf * ρ2_ρ1
        params.T[i, j] = inputs.T_inf * T2_T1

        # Post-shock velocity components (correct orientation)
        Vn1 = W_inf * sin(β)
        Vt1 = W_inf * cos(β)
        Vn2 = Vn1 / ρ2_ρ1
        Vt2 = Vt1

        # Transform to global coordinates
        params.u[i, j] = Vn2 * cos(β) - Vt2 * sin(β)
        params.v[i, j] = Vn2 * sin(β) + Vt2 * cos(β)

        # Log variables
        params.R[i, j] = log(params.ρ[i, j])
        params.ψ[i, j] = log(params.p[i, j]) - inputs.γ * params.R[i, j]
        params.P[i, j] = log(params.p[i, j])
    end

    # --- Step 2: Update shock position (simplified Anderson) ---
    for i in 2:(inputs.Ny - 1)
        β = β_func(i, grid.xs, y_profile)
        if abs(y_profile[i]) < 0.05
            β = π/2
        end

        # Local shock speed W_local
        M1n = inputs.M_inf * sin(β)
        denom = max((inputs.γ - 1) * M1n^2 + 2, 1e-12)
        ρ2ρ1 = ((inputs.γ + 1) * M1n^2) / denom
        Vn2 = W_inf * sin(β) / ρ2ρ1
        W_local = W_inf * cos(β) - Vn2

        Δx = clamp(-0.05 * W_local * dt,
                   -0.5 * abs(grid.xs[i] - grid.xb[i]),
                    0.5 * abs(grid.xs[i] - grid.xb[i]))
        grid.xs[i] += Δx
    end

    # --- Step 3: Smooth shock shape ---
    xs_new = copy(grid.xs)
    for i in 2:(inputs.Ny - 1)
        xs_new[i] = 0.25 * grid.xs[i-1] + 0.5 * grid.xs[i] + 0.25 * grid.xs[i+1]
    end
    grid.xs .= xs_new

    # --- Step 4: Remap Z after shock move ---
    for i in 1:inputs.Ny
        Δ = grid.xs[i] - grid.xb[i]
        for jj in 1:inputs.Nx
            grid.Z[i, jj] = (grid.x_grid[i, jj] - grid.xb[i]) / Δ
        end
    end

    return params, grid
end

function body_boundary(params, grid, inputs, derivatives, corrector, dt)
    j = 1   # body boundary column index
    dZ = 1 / (inputs.Nx - 1)
    dy = 1 / (inputs.Ny - 1)

    # Freestream sound speed (for finite wave adjustment)
    a_inf = sqrt(inputs.γ * inputs.R_gas * inputs.T_inf)

    for i in 2:(inputs.Ny - 1)
        delta = grid.xs[i] - grid.xb[i]

        # --- Local surface slope (for normal/tangent) ---
        if i == inputs.Ny - 1
            dyb = grid.y_grid[i, j] - grid.y_grid[i-1, j]
            dxb = grid.xb[i] - grid.xb[i-1]
        else
            dyb = grid.y_grid[i+1, j] - grid.y_grid[i, j]
            dxb = grid.xb[i+1] - grid.xb[i]
        end
        t_mag = sqrt(dxb^2 + dyb^2)
        tx, ty = dxb / t_mag, dyb / t_mag
        nx, ny = -ty, tx  # outward normal

        # --- Predictor step (forward differences) ---
        C = C_func(grid.Z[i, j], i, j, inputs.Ny, grid.y_grid, grid.xs, grid.xb)
        B = B_func(params.u[i, j], grid.W[i], grid.Z[i, j], params.v[i, j], C, i, j, delta)

        dudZ = (params.u[i, j+1] - params.u[i, j]) / dZ
        dvdZ = (params.v[i, j+1] - params.v[i, j]) / dZ
        dPdZ = (params.P[i, j+1] - params.P[i, j]) / dZ
        dRdZ = (params.R[i, j+1] - params.R[i, j]) / dZ
        dψdZ = (params.ψ[i, j+1] - params.ψ[i, j]) / dZ
        dpdZ = params.p[i, j] * dPdZ

        if i == inputs.Ny - 1
            dudy = (params.u[i, j] - params.u[i-1, j]) / dy
            dvdy = (params.v[i, j] - params.v[i-1, j]) / dy
            dPdy = (params.P[i, j] - params.P[i-1, j]) / dy
            dRdy = (params.R[i, j] - params.R[i-1, j]) / dy
            dψdy = (params.ψ[i, j] - params.ψ[i-1, j]) / dy
        else
            dudy = (params.u[i+1, j] - params.u[i, j]) / dy
            dvdy = (params.v[i+1, j] - params.v[i, j]) / dy
            dPdy = (params.P[i+1, j] - params.P[i, j]) / dy
            dRdy = (params.R[i+1, j] - params.R[i, j]) / dy
            dψdy = (params.ψ[i+1, j] - params.ψ[i, j]) / dy
        end

        derivatives.dudt[i, j] = -(B*dudZ + params.v[i, j]*dudy +
                                   (params.p[i, j]/(params.ρ[i, j]*delta))*dpdZ)
        derivatives.dvdt[i, j] = -(B*dvdZ + params.v[i, j]*dvdy +
                                   (params.p[i, j]*C)/(params.ρ[i, j]*delta)*dPdZ +
                                   (params.p[i, j]/params.ρ[i, j])*dPdy)
        derivatives.dRdt[i, j] = -(B*dRdZ + (1/delta)*dudZ +
                                   (C/delta)*dvdZ + dvdy + params.v[i, j]*dRdy)
        derivatives.dψdt[i, j] = -(B*dψdZ + params.v[i, j]*dψdy)

        # Predictor update
        corrector.u_[i, j] = params.u[i, j] + derivatives.dudt[i, j]*dt
        corrector.v_[i, j] = params.v[i, j] + derivatives.dvdt[i, j]*dt
        corrector.R_[i, j] = params.R[i, j] + derivatives.dRdt[i, j]*dt
        corrector.ψ_[i, j] = params.ψ[i, j] + derivatives.dψdt[i, j]*dt

        corrector.ρ_[i, j] = exp(corrector.R_[i, j])
        corrector.p_[i, j] = exp(corrector.ψ_[i, j] + inputs.γ * corrector.R_[i, j])
        corrector.P_[i, j] = log(max(corrector.p_[i, j], 1e-12))
        corrector.T_[i, j] = corrector.p_[i, j] / (corrector.ρ_[i, j] * inputs.R_gas)

        # --- Corrector step (forward differences again) ---
        C_ = C_func(grid.Z[i, j], i, j, inputs.Ny, grid.y_grid, grid.xs, grid.xb)
        B_ = B_func(corrector.u_[i, j], grid.W[i], grid.Z[i, j],
                    corrector.v_[i, j], C_, i, j, delta)

        dudZ_ = (corrector.u_[i, j+1] - corrector.u_[i, j]) / dZ
        dvdZ_ = (corrector.v_[i, j+1] - corrector.v_[i, j]) / dZ
        dPdZ_ = (corrector.P_[i, j+1] - corrector.P_[i, j]) / dZ
        dRdZ_ = (corrector.R_[i, j+1] - corrector.R_[i, j]) / dZ
        dψdZ_ = (corrector.ψ_[i, j+1] - corrector.ψ_[i, j]) / dZ
        dpdZ_ = corrector.p_[i, j] * dPdZ_

        if i == inputs.Ny - 1
            dudy_ = (corrector.u_[i, j] - corrector.u_[i-1, j]) / dy
            dvdy_ = (corrector.v_[i, j] - corrector.v_[i-1, j]) / dy
            dPdy_ = (corrector.P_[i, j] - corrector.P_[i-1, j]) / dy
            dRdy_ = (corrector.R_[i, j] - corrector.R_[i-1, j]) / dy
            dψdy_ = (corrector.ψ_[i, j] - corrector.ψ_[i-1, j]) / dy
        else
            dudy_ = (corrector.u_[i+1, j] - corrector.u_[i, j]) / dy
            dvdy_ = (corrector.v_[i+1, j] - corrector.v_[i, j]) / dy
            dPdy_ = (corrector.P_[i+1, j] - corrector.P_[i, j]) / dy
            dRdy_ = (corrector.R_[i+1, j] - corrector.R_[i, j]) / dy
            dψdy_ = (corrector.ψ_[i+1, j] - corrector.ψ_[i, j]) / dy
        end

        derivatives.dudt_[i, j] = -(B_*dudZ_ + corrector.v_[i, j]*dudy_ +
                                    (corrector.p_[i, j]/(corrector.ρ_[i, j]*delta))*dpdZ_)
        derivatives.dvdt_[i, j] = -(B_*dvdZ_ + corrector.v_[i, j]*dvdy_ +
                                    (corrector.p_[i, j]*C_)/(corrector.ρ_[i, j]*delta)*dPdZ_ +
                                    (corrector.p_[i, j]/corrector.ρ_[i, j])*dPdy_)
        derivatives.dRdt_[i, j] = -(B_*dRdZ_ + (1/delta)*dudZ_ +
                                    (C_/delta)*dvdZ_ + dvdy_ + corrector.v_[i, j]*dRdy_)
        derivatives.dψdt_[i, j] = -(B_*dψdZ_ + corrector.v_[i, j]*dψdy_)

        # --- Average derivatives ---
        dudt_avg = 0.5 * (derivatives.dudt[i, j] + derivatives.dudt_[i, j])
        dvdt_avg = 0.5 * (derivatives.dvdt[i, j] + derivatives.dvdt_[i, j])
        dRdt_avg = 0.5 * (derivatives.dRdt[i, j] + derivatives.dRdt_[i, j])
        dψdt_avg = 0.5 * (derivatives.dψdt[i, j] + derivatives.dψdt_[i, j])

        # --- Update velocities (preliminary) ---
        u_new = params.u[i, j] + dudt_avg * dt
        v_new = params.v[i, j] + dvdt_avg * dt

        # --- Enforce tangency: remove normal velocity ---
        Vn = u_new * nx + v_new * ny
        u_new -= Vn * nx
        v_new -= Vn * ny

        params.u[i, j] = u_new
        params.v[i, j] = v_new

        # --- Update R and ψ (before pressure correction) ---
        params.R[i, j] += dRdt_avg * dt
        params.ψ[i, j] += dψdt_avg * dt

        # --- Compute base p, ρ from updated R, ψ ---
        params.ρ[i, j] = exp(params.R[i, j])
        params.p[i, j] = exp(params.ψ[i, j] + inputs.γ * params.R[i, j])

        # --- Finite wave adjustment (Anderson Eq. 5.53 concept) ---
        Δp = -params.ρ[i, j] * Vn * a_inf
        Δp = clamp(Δp, -0.5 * params.p[i, j], 0.5 * params.p[i, j])  # prevent negative or extreme p changes
        params.p[i, j] += Δp

        # --- Recompute density using previous T before updating T ---
        params.ρ[i, j] = params.p[i, j] / (inputs.R_gas * params.T[i, j])

        # --- Update derived variables with safe log ---
        params.P[i, j] = log(max(params.p[i, j], 1e-12))
        params.T[i, j] = params.p[i, j] / (params.ρ[i, j] * inputs.R_gas)
    end

    return params
end

function downstream_boundary(params, grid, inputs)
    Ny, Nx = inputs.Ny, inputs.Nx

    for j in 1:Nx
        params.u[Ny, j] = 2*params.u[Ny-1, j] - params.u[Ny-2, j]
        params.v[Ny, j] = 2*params.v[Ny-1, j] - params.v[Ny-2, j]
        params.p[Ny, j] = 2*params.p[Ny-1, j] - params.p[Ny-2, j]
        params.ρ[Ny, j] = 2*params.ρ[Ny-1, j] - params.ρ[Ny-2, j]
        params.T[Ny, j] = 2*params.T[Ny-1, j] - params.T[Ny-2, j]
        params.P[Ny, j] = 2*params.P[Ny-1, j] - params.P[Ny-2, j]
        params.R[Ny, j] = 2*params.R[Ny-1, j] - params.R[Ny-2, j]
        params.ψ[Ny, j] = 2*params.ψ[Ny-1, j] - params.ψ[Ny-2, j]
    end

    return params
end

function centerline_boundary(params)
    Ny, Nx = size(params.p)

    # Scalar fields: even symmetry across centerline
    params.p[1, :] .= params.p[2, :]
    params.T[1, :] .= params.T[2, :]
    params.ρ[1, :] .= params.ρ[2, :]
    params.P[1, :] .= params.P[2, :]
    params.R[1, :] .= params.R[2, :]
    params.ψ[1, :] .= params.ψ[2, :]

    # Velocity: normal component (v) = 0, tangential (u) mirrored
    params.u[1, :] .= params.u[2, :]   # tangential is symmetric
    params.v[1, :] .= 0.0              # normal is antisymmetric
end

function plot_flowfield(params, grid, inputs, derivatives, corrector, dt)
    try
        # --- Speed of sound at freestream ---
        a_inf = sqrt(inputs.γ * inputs.R_gas * inputs.T_inf)

        # --- Flow quantities ---
        Vmag = sqrt.(params.u.^2 .+ params.v.^2)
        Mmag = Vmag ./ a_inf

        # --- Flatten arrays (upper half solution) ---
        x_flat = vec(grid.x_grid)
        y_flat = vec(grid.y_grid)
        M_flat = vec(Mmag)
        u_flat = vec(params.u)
        v_flat = vec(params.v)
        T_flat = vec(params.T)
        p_flat = vec(params.p)
        ρ_flat = vec(params.ρ)

        # --- Mirror about x-axis ---
        x_mirror = copy(x_flat)
        y_mirror = -y_flat
        M_mirror = copy(M_flat)
        u_mirror = copy(u_flat)
        v_mirror = -v_flat
        T_mirror = copy(T_flat)
        p_mirror = copy(p_flat)
        ρ_mirror = copy(ρ_flat)

        # --- Combine top and bottom halves ---
        x_all = vcat(x_flat, x_mirror)
        y_all = vcat(y_flat, y_mirror)
        M_all = vcat(M_flat, M_mirror)
        p_all = vcat(p_flat, p_mirror)
        u_all = vcat(u_flat, u_mirror)
        v_all = vcat(v_flat, v_mirror)
        ρ_all = vcat(ρ_flat, ρ_mirror)
        T_all = vcat(T_flat, T_mirror)

        # --- Custom colormap (your HSV variant) ---
        trunc_hsv = [get(ColorSchemes.hsv, f) for f in range(0, stop=0.67, length=256)]
        reversed_trunc_hsv = reverse(trunc_hsv)
        custom_cmap = cgrad(reversed_trunc_hsv)

        # --- Plot flowfield as colored scatter ---
        plt = scatter(
            x_all,
            y_all,
            marker_z = M_all,
            colorbar = true,
            xlabel = "x (m)",
            ylabel = "y (m)",
            title = "Mach number magnitude",
            c = custom_cmap,        # <-- Use custom colormap
            markersize = 3,
            aspect_ratio = :equal,
            legend = false,
            size = (1200, 900),
        )

        # --- Body outline (full circle) ---
        θ = range(0, 2π, length=400)
        x_body = -inputs.R_body * cos.(θ)
        y_body = inputs.R_body * sin.(θ)
        plot!(x_body, y_body, lw=3, c=:black)

        display(plt)

        return (x_all, y_all, M_all)
    catch e
        @warn "Plotting failed!" exception = (e, catch_backtrace())
    end
end

function reform_grid(params, grid, inputs)
    Ny, Nx = size(grid.x_grid)

    for i in 1:Ny
        xb_i = grid.xb[i]
        xs_i = grid.xs[i]
        Δx = xs_i - xb_i

        # --- Recompute computational coordinate Z for this row ---
        # Ensures Z always maps from body (0) to shock (1) even if xs moves
        for j in 1:Nx
            grid.Z[i, j] = (grid.x_grid[i, j] - xb_i) / Δx
        end

        # --- Remap physical x_grid using updated shock position ---
        for j in 1:Nx
            grid.x_grid[i, j] = xb_i + grid.Z[i, j] * Δx
        end
    end

    return grid
end

# Main Simulation Loop ########################################################################################################
function run_simulation(; M_inf, T_inf, p_inf, ρ_inf, γ, R_gas, Nx, Ny, CFL, R_body, max_steps)
    params = initialize_params(Nx, Ny, ρ_inf, p_inf, M_inf, T_inf, γ, R_gas)
    grid = initialize_GridVars(Nx, Ny)
    inputs = initialize_inputs(γ, R_gas, CFL, Nx, Ny, M_inf, T_inf, p_inf, ρ_inf, R_body)
    derivatives = initialize_derivatives(Nx, Ny)
    corrector = initialize_corrector(Nx, Ny)
    grid.Z = transform_grid(params, grid, inputs)
    grid.x_grid, grid.y_grid, grid.xb, grid.xs, grid.Z, grid.W = form_domain(params, grid, inputs)
    steps = 0
    plot_flowfield(params, grid, inputs, derivatives, corrector, 0.0)  # Initial plot
    
    while steps < max_steps
        dt = update_dt(params, grid, inputs) 
        interior_points(params, grid, inputs, derivatives, corrector, dt)

        #shock_boundary(params, grid, inputs, dt)  # < - something is making xs NaN
        body_boundary(params, grid, inputs, derivatives, corrector, dt)
        downstream_boundary(params, grid, inputs)
        centerline_boundary(params)

        reform_grid(params, grid, inputs)
    
        if steps % 1000 == 0
            println("Time step: ", steps)
            plot_flowfield(params, grid, inputs, derivatives, corrector, dt)
        end
        steps += 1
    end
    
end

# MAIN #
run_simulation(M_inf=5.0, T_inf=273.15, p_inf=101325, ρ_inf = 1.225, γ=1.4, R_gas=287.05, Nx=20, Ny=50, CFL=0.00001, R_body=1.0, max_steps=1000000)
