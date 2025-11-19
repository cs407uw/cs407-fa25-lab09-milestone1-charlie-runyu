package com.cs407.lab09

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.paint
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.cs407.lab09.R
import com.cs407.lab09.ui.theme.Lab09Theme
import kotlin.math.roundToInt

// Main Activity
class MainActivity : ComponentActivity() {

    private val viewModel: BallViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Lab09Theme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    GameScreen(viewModel = viewModel)
                }
            }
        }
    }
}

@Composable
fun GameScreen(viewModel: BallViewModel) {
    // TODO: Initialize the sensorManager
    // ---- SensorManager ----
    val context = LocalContext.current

    val sensorManager = remember {
        // ... getSystemService ...
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    }

    // TODO: Get the gravitySensor
    val gravitySensor = remember {
        // ... getDefaultSensor ...
        //sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    }

    // This effect runs when the composable enters the screen
    // and cleans up when it leaves
    DisposableEffect(sensorManager, gravitySensor) {
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                // TODO: Pass the sensor event to the ViewModel
                event?.let {
                    viewModel.onSensorDataChanged(it)
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
                    // Do nothing
            }
        }

            // TODO: Register the sensor listener
            // (Don't forget to add a null check for gravitySensor!)
        if (gravitySensor != null)
        {
                sensorManager.registerListener(
                    listener,
                    gravitySensor,
                    SensorManager.SENSOR_DELAY_GAME
                )
        }

            // onDispose is called when the composable leaves the screen
        onDispose{
                // TODO: Unregister the sensor listener
                // (Don't forget to add a null check for gravitySensor!)
            if (gravitySensor != null) {
                    sensorManager.unregisterListener(listener)
            }
        }
    }

    // UI layout
    Column(modifier = Modifier.fillMaxSize()) {
        // 1. The Reset Button
        Button(
            onClick = {
                // TODO: Call the reset function on the ViewModel
                viewModel.reset()
            },
            modifier = Modifier
                .align(Alignment.CenterHorizontally)
                .padding(16.dp)
        ) {
            Text(text = "Reset")
        }

        // 2. The Game Field
        val ballSize = 50.dp
        val ballSizePx = with(LocalDensity.current) { ballSize.toPx() }

        // TODO: Collect the ball's position from the ViewModel
         val ballPosition by viewModel.ballPosition.collectAsStateWithLifecycle()

        Box(
            modifier = Modifier
                .fillMaxSize()
                .weight(1f)
                .paint(
                    painter = painterResource(id = R.drawable.field),
                    contentScale = ContentScale.FillBounds
                )
                .onSizeChanged { size ->
                    // TODO: Tell the ViewModel the size of the field
                    viewModel.initBall(size.width.toFloat(), size.height.toFloat(), ballSizePx)
                }
        ) {
            // 3. The Ball
            Image(
                painter = painterResource(id = R.drawable.soccer),
                contentDescription = "Soccer Ball",
                modifier = Modifier
                    .size(ballSize)
                    .offset {
                        // TODO: Use the collected ballPosition to set the offset
                        // Hint: You need to convert Float to Int
                        IntOffset(
                            x = ballPosition.x.roundToInt(),
                            y = ballPosition.y.roundToInt()
                        )
                    }
            )
        }
    }
}
