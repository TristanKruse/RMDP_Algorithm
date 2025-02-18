<!-- Press Ctrl+Shift+V for preview -->

# Route Processing Flow
 
## Process all routes:

```mermaid

stateDiagram-v2
    [*] --> ProcessAllRoutes: Start timestep
    
    state ProcessAllRoutes {
        [*] --> VehicleCheck: Get next vehicle
        VehicleCheck --> IdleHandling: No route/current_phase
        VehicleCheck --> StopProcessing: Has route

        state IdleHandling {
            [*] --> UpdateIdleMetrics
            UpdateIdleMetrics --> ProcessIdleMovement: No current_phase
        }

        state StopProcessing {
            [*] --> GetFirstStop: Extract node_id, pickups, deliveries
            GetFirstStop --> PickupHandling: Has pickups
            GetFirstStop --> DeliveryHandling: Has deliveries

            state PickupHandling {
                [*] --> CheckPhase
                CheckPhase --> InitializePhase: No phase
                CheckPhase --> ProcessMovement: Has phase
                InitializePhase --> ProcessMovement: Set target
                
                state ProcessMovement {
                    [*] --> UpdateProgress: Increment progress
                    UpdateProgress --> CalculateNewPosition
                    CalculateNewPosition --> CheckArrival
                    CheckArrival --> StillMoving: progress < 1.0
                    CheckArrival --> HandleArrival: progress >= 1.0
                    HandleArrival --> CheckReadyTime
                    CheckReadyTime --> WaitAtRestaurant: Not ready
                    CheckReadyTime --> StartService: Ready
                    StartService --> ServiceProgress
                }
            }

            state DeliveryHandling {
                [*] --> CheckDeliveryPhase
                CheckDeliveryPhase --> InitDeliveryPhase: No phase
                CheckDeliveryPhase --> ProcessDeliveryMovement: Has phase
                InitDeliveryPhase --> ProcessDeliveryMovement: Set target
                
                state ProcessDeliveryMovement {
                    [*] --> UpdateDeliveryProgress
                    UpdateDeliveryProgress --> CalculateDeliveryPosition
                    CalculateDeliveryPosition --> CheckDeliveryArrival
                    CheckDeliveryArrival --> StillDeliverying: progress < 1.0
                    CheckDeliveryArrival --> StartDeliveryService: progress >= 1.0
                    StartDeliveryService --> DeliveryServiceProgress
                }
            }
        }

        ProcessIdleMovement --> NextVehicle
        StillMoving --> NextVehicle
        WaitAtRestaurant --> NextVehicle
        ServiceProgress --> NextVehicle
        StillDeliverying --> NextVehicle
        DeliveryServiceProgress --> NextVehicle
        NextVehicle --> [*]: Update metrics for timestep
    }

    ProcessAllRoutes --> [*]: End timestep

```


## Process a single order

``` mermaid

stateDiagram-v2
    [*] --> ValidateOrder: Initialize & get order
    ValidateOrder --> [*]: Order not found
    
    ValidateOrder --> CheckRerouting: Order found
    
    state CheckRerouting {
        [*] --> ReroutingCheck: Check if rerouting needed
        ReroutingCheck --> ResetPhase: Different order in pickup
        ResetPhase --> InitPhase: Reset & make order pending
        ReroutingCheck --> InitPhase: No rerouting needed
    }
    
    state ProcessPhase {
        InitPhase --> CheckService: Initialize phase if needed
        CheckService --> ServiceHandling: Is servicing
        CheckService --> MovementHandling: Not servicing
        
        state ServiceHandling {
            [*] --> ProcessService
            ProcessService --> CompleteService: Service done
            ProcessService --> ContinueService: Service ongoing
        }
        
        state MovementHandling {
            [*] --> UpdateProgress: Handle movement
            UpdateProgress --> CalculateNewPosition
            CalculateNewPosition --> CheckArrival: Check arrival
            CheckArrival --> HandleArrival: progress >= 1.0
            CheckArrival --> ContinueMovement: progress < 1.0
        }
    }
    
    ServiceHandling --> ReturnState
    MovementHandling --> ReturnState
    
    state ReturnState {
        [*] --> ReturnValues: Return appropriate values
        ReturnValues --> CompleteReturn: (new_loc, distance, delay, completed)
    }
    
    ReturnState --> [*]

 ```