// Code by Alexandre Haddad-Delaveau

[Setting name="Port"]
string port = "6969";

[Setting name="Allow Replays"]
bool replaysAllowed = false;

// Car Info
float speed = 0;
float rpm = 0;
int gear = 0;

// Car Position (used for training)
float posX = 0;
float posY = 0;
float posZ = 0;

// Game State
bool racingState = false;

void Main()
{
    // Repeat every 10ms
    while (true) {
        updateGameInfo();
        notifyServer();
        sleep(20);
    }
}

void updateGameInfo()
{
    // Make sure we are in a game
    if (GetApp().CurrentPlayground is null && !replaysAllowed) {
        resetValues();
        return;
    }

    // Get race and player data
    auto RaceData = MLFeed::GetRaceData_V4();
    auto player = RaceData.GetPlayer_V4(MLFeed::LocalPlayersName);

    // Make sure player is playing
    if (player is null && !replaysAllowed) {
        resetValues();
        return;
    }

    // Check if player is spawned
    if (replaysAllowed || player.SpawnStatus == 2) {
        // If spawned, update values
        racingState = true;

        // Attempt to get player state
        auto vehicleVisState = VehicleState::ViewingPlayerState();
        if (vehicleVisState is null) {
            resetValues();
            print("VEHICLE NULL");
            return;
        }

        // Update values
        vec3 pos = vehicleVisState.Position;
        posX = pos[0];
        posY = pos[1];
        posZ = pos[2];
        speed = vehicleVisState.FrontSpeed;
        rpm = VehicleState::GetRPM(vehicleVisState);
        gear = vehicleVisState.CurGear;
        
    } else {
        // If not spawned, reset values
        resetValues();
        return;
    }

}

void resetValues() {
    posX = 0;
    posY = 0;
    posZ = 0;
    speed = 0;
    rpm = 0;
    gear = 0;
    racingState = false;
}

void notifyServer() {
    // Convert to json
    dictionary dictValues = {{'posX', posX}, {'posY', posY}, {'posZ', posZ}, {'speed', speed}, {'gear', gear}, {'rpm', rpm}, {'racing', racingState}};
    Json::Value@ jsonValues = dictValues.ToJson();

    // Send Request
    string url = "http://127.0.0.1:" + port;
    Net::HttpRequest@ request = Net::HttpPost(url, Json::Write(jsonValues));
    print(Json::Write(jsonValues));
}