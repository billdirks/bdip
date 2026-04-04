# Architectural Component Diagram

The following diagram provides a top-down view of the strict module separation inside the `bdip` workspace. It illustrates the data boundary isolating the **Application Binary (UI & CLI routing)** from the **Core Processing Library (GPU execution & Data I/O)**.

```mermaid 
flowchart TB
    %% Entities
    User((User))
    FileSystem[(File System)]

    %% Application Binary Workspace
    subgraph AppBinary ["📦 Application Binary (bdip)"]
        CLI["Terminal CLI Router<br/>(Parses Arguments)"]
        
        subgraph UILayer ["🖥 UI Presentation Layer (Iced)"]
            UIEvents["Event Handler<br/>(Cmd+O / Cmd+S / Sliders)"]
            CanvasViewer["Image Canvas<br/>(Displays GPU Texture)"]
            UIEvents --> CanvasViewer
        end
    end

    %% Core Library Workspace
    subgraph CoreLib ["📦 Core Processing Library (bdip_core)"]
        IO["I/O & Serialization Manager<br/>(Reads/Writes JPG, TIFF, GIF)"]
        
        subgraph StateManager ["🧠 State & History Manager"]
            UndoRedo["Undo/Redo Buffer Queues"]
            TransformStack["Transformation Sequence Stack<br/>(List of active filters)"]
            UndoRedo <--> TransformStack
        end

        subgraph GPUEngine ["⚙️ GPU Transformation Engine (wgpu)"]
            HardwareSetup["GPU Interface<br/>(Instance, Adapter, Device)"]
            WGSLShaders["Processing Logic<br/>(WGSL Shaders: Brightness, etc.)"]
            HardwareSetup --- WGSLShaders
        end
    end

    %% Human Inputs
    User -->|Terminal Commands| CLI
    User -->|Mouse/Keyboard Interactions| UIEvents

    %% Inter-Component Flow
    CLI -->|Load on Startup| IO
    UIEvents -->|Request File Open/Save| IO
    UIEvents -->|Dispatch Filter Changes| TransformStack

    IO -->|Uploads Raw Pixels| GPUEngine
    IO <-->|Persists to Disk| FileSystem

    TransformStack -->|Submit Parameters| WGSLShaders
    GPUEngine -->|Renders Output Texture| CanvasViewer
    GPUEngine -->|Extracts Finished Frame| IO
```
