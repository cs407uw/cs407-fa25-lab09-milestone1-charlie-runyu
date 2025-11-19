package com.cs407.lab09

/**
 * Represents a ball that can move. (No Android UI imports!)
 *
 * Constructor parameters:
 * - backgroundWidth: the width of the background, of type Float
 * - backgroundHeight: the height of the background, of type Float
 * - ballSize: the width/height of the ball, of type Float
 */
class Ball(
    private val backgroundWidth: Float,
    private val backgroundHeight: Float,
    private val ballSize: Float
) {
    var posX = 0f
    var posY = 0f
    var velocityX = 0f
    var velocityY = 0f
    private var accX = 0f
    private var accY = 0f

    private var isFirstUpdate = true

    init {
        // TODO: Call
        reset()
    }

    /**
     * Updates the ball's position and velocity based on the given acceleration and time step.
     * (See lab handout for physics equations)
     */
    fun updatePositionAndVelocity(xAcc: Float, yAcc: Float, dT: Float) {
        if (dT <= 0f) return
        if(isFirstUpdate) {
            isFirstUpdate = false
            accX = xAcc
            accY = yAcc
            return
        }
        // Save old acceleration
        val a0x = accX
        val a0y = accY
        val a1x = xAcc
        val a1y = yAcc


        // ---- X velocity ----
        val newVx = velocityX + 0.5f * (a0x + a1x) * dT
        // ---- X displacement ----
        val dx = velocityX * dT + (1f / 6f) * dT * dT * (3f * a0x + a1x)

        // ---- Y velocity ----
        val newVy = velocityY + 0.5f * (a0y + a1y) * dT
        // ---- Y displacement ----
        val dy = velocityY * dT + (1f / 6f) * dT * dT * (3f * a0y + a1y)

        // Update
        posX += dx
        posY += dy
        velocityX = newVx
        velocityY = newVy
        accX = a1x
        accY = a1y

        checkBoundaries()
    }

    /**
     * Ensures the ball does not move outside the boundaries.
     * When it collides, velocity and acceleration perpendicular to the
     * boundary should be set to 0.
     */
    fun checkBoundaries() {
        // TODO: implement the checkBoundaries function
        // (Check all 4 walls: left, right, top, bottom)

        val radius = ballSize / 2f

        // Left
        if (posX < 0f) {
            posX = 0f
            velocityX = 0f
            accX = 0f
        }

        // Right
        if (posX + ballSize > backgroundWidth) {
            posX = backgroundWidth - ballSize
            velocityX = 0f
            accX = 0f
        }

        // Top
        if (posY < 0f) {
            posY = 0f
            velocityY = 0f
            accY = 0f
        }

        // Bottom
        if (posY + ballSize > backgroundHeight) {
            posY = backgroundHeight - ballSize
            velocityY = 0f
            accY = 0f
        }

    }

    /**
     * Resets the ball to the center of the screen with zero
     * velocity and acceleration.
     */
    fun reset() {
        // TODO: implement the reset function
        // (Reset posX, posY, velocityX, velocityY, accX, accY, isFirstUpdate)

        posX = (backgroundWidth - ballSize) / 2f
        posY = (backgroundHeight - ballSize) / 2f
        velocityX = 0f
        velocityY = 0f
        accX = 0f
        accY = 0f
        isFirstUpdate = true
    }
}