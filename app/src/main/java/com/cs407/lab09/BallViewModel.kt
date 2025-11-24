package com.cs407.lab09

import android.hardware.Sensor
import android.hardware.SensorEvent
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class BallViewModel : ViewModel() {

    private var ball: Ball? = null
    private var lastTimestamp: Long = 0L

    // Expose the ball's position as a StateFlow
    private val _ballPosition = MutableStateFlow(Offset.Zero)
    val ballPosition: StateFlow<Offset> = _ballPosition.asStateFlow()

    /**
     * Called by the UI when the game field's size is known.
     */
    fun initBall(fieldWidth: Float, fieldHeight: Float, ballSizePx: Float) {
        if (ball == null) {
            // TODO: Initialize the ball instance
            ball = Ball(fieldWidth, fieldHeight, ballSizePx)
            // TODO: Update the StateFlow with the initial position
            _ballPosition.value = Offset(ball!!.posX, ball!!.posY)
        }

    }

    /**
     * Called by the SensorEventListener in the UI.
     */
    fun onSensorDataChanged(event: SensorEvent) {
        // Ensure ball is initialized
        val currentBall = ball ?: return

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {//replaced TYPE_GRAVITY
            if (lastTimestamp != 0L) {
                // TODO: Calculate the time difference (dT) in seconds
                // Hint: event.timestamp is in nanoseconds
                //val NS2S = 1.0f / 1000000000.0f
                // val dT = ...
                val NS2S = 1f / 1_000_000_000f
                val dT = (event.timestamp - lastTimestamp) * NS2S

                // Sensor axes: +x=right, +y=up
                // Screen axes: +x=right, +y=down → flip y
                val rawX = event.values[0]
                val rawY = event.values[1]
                val ax = -rawX       //conventional right - right
                val ay = rawY        //can be -event.values[1] if it should be opposite

                // TODO: Update the ball's position and velocity
                // Hint: The sensor's x and y-axis are inverted
                // currentBall.updatePositionAndVelocity(xAcc = ..., yAcc = ..., dT = ...)
                currentBall.updatePositionAndVelocity(ax, ay, dT)


                // TODO: Update the StateFlow to notify the UI
                // _ballPosition.update { Offset(currentBall.posX, currentBall.posY) }
                _ballPosition.update {
                    Offset(currentBall.posX, currentBall.posY)
                }

            }

            // TODO: Update the lastTimestamp
             lastTimestamp = event.timestamp
        }
    }

    fun reset() {
        // TODO: Reset the ball's state
         ball?.reset()

        // TODO: Update the StateFlow with the reset position
        ball?.let { _ballPosition.value = Offset(it.posX, it.posY) }

        // TODO: Reset the lastTimestamp
         lastTimestamp = 0L
    }
}