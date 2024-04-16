package com.dicoding.picodiploma.mycamera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.health.connect.datatypes.HeightRecord
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.dicoding.picodiploma.mycamera.ml.EfficientdetLite0V1.DetectionResult
import org.tensorflow.lite.task.gms.vision.detector.Detection
import java.text.NumberFormat
import java.util.LinkedList
import kotlin.math.max

class OverlayView @JvmOverloads constructor(context: Context, attr: AttributeSet? = null): View(context, attr){
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var results: List<Detection> = LinkedList<Detection>()
    private var scaleFactor: Float = 1f

    private var bounds = Rect()

    init {
       iniPaints()
    }

    private fun iniPaints() {
        boxPaint.color = ContextCompat.getColor(context,R.color.bounding_box_color)
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 8f

        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f
    }

    //memasukkan nilai dari hasil inferensi di Activity ke dalam OverlayView
    fun setResult(detectionResults: MutableList<Detection>, imgHeight: Int, imgWidth: Int){
        results = detectionResults
        scaleFactor = max(width * 1f / imgWidth, height * 1f / imgHeight) // ukuran kotak detectionya
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (result in results){
            val boundingBox = result.boundingBox
            val left = boundingBox.left * scaleFactor
            val top = boundingBox.top * scaleFactor
            val right = boundingBox.right * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor

            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display alongside detected objects
            val drawableText = "${result.categories[0].label} " +
                    NumberFormat.getPercentInstance().format(result.categories[0].score)

            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }
    // clear bounding
    fun clear() {
        boxPaint.reset()
        textBackgroundPaint.reset()
        textPaint.reset()
        invalidate()
        iniPaints()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}