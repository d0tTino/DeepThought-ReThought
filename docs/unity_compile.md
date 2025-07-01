# Compiling C# Scripts in Unity

The repository ships a few helper scripts for NATS connectivity. To use them in Unity you simply add the `.cs` files to your project's `Assets` folder. Unity automatically compiles any C# files it finds when the editor gains focus.

## Steps

1. Open your Unity project in the editor.
2. Copy `NatsService.cs`, `NatsJetStreamManager.cs`, `NatsMessageExample.cs` and `UnityMainThreadDispatcher.cs` into `Assets/Scripts/` (or another folder under `Assets/`).
3. Unity triggers compilation automatically. Check the **Console** window for errors.
4. Once compilation succeeds, attach the scripts to GameObjects in your scene and assign any required parameters in the Inspector.

### Command-Line Build

You can also verify the scripts compile from the command line without opening the editor:

```bash
/Path/To/Unity -batchmode -nographics -quit \
    -projectPath /path/to/YourProject -logFile -
```

Unity will compile the scripts and exit. Any compilation errors are printed to the console.

### Manual Testing

1. Create an empty GameObject and add the `NatsService` component.
2. Enter Play mode. If no console errors appear the scripts compiled correctly and the NATS connection will attempt to start.

