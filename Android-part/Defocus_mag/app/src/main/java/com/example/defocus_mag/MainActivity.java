package com.example.defocus_mag;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import com.pnikosis.materialishprogress.ProgressWheel;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private static final String MODEL_FILE1 = "file:///android_asset/DeblurrGAN_last_opt.pb";
    private static final String MODEL_FILE2 = "file:///android_asset/defocus_transformed_quantized_last.pb";
    private static final String MODEL_FILE3 = "file:///android_asset/DeblurrGAN_last_opt.pb";
    private static final String MODEL_FILE4 = "file:///android_asset/Refocus_transformed_quantized_last.pb";

    private static final int RESULT_LOAD_IMAGE=1;

    //have to add original names
//    private static final String INPUT_NODE = "image_feed";
//    private static final String OUTPUT_NODE = "generator_1/deprocess/truediv";
    private static final String INPUT_NODE = "sub:0";
    private static final String MAG_NODE = "magnification:0";
    private static final String DELTA_NODE = "delta:0";
    private static final String OUTPUT_NODE = "generator/clip_by_value:0";

    private static final int WANTED_WIDTH = 256;
    private static final int WANTED_HEIGHT = 256;

    private Button mButtonUpload;
    private ImageButton mButtonSave;
    private Button mButtonDefocus, mButtonDeblur;
    private ImageView mImageView;
    private ImageView mImageView_f;
    private Bitmap mGeneratedBitmap, mGeneratedBitmap_f;
    private Switch mdefocusSwitch, mrefocusSwitch;
    private TextView defocus_param,refocus_param;
    private Bitmap mFocusMeasure;

    private Bitmap bitmap_orig;

    private FrameLayout progressOverlay;
    private ProgressBar progressWheel;

    private TensorFlowInferenceInterface mInferenceInterface;

    private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("Hello", "OpenCV loaded successfully");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        progressOverlay = findViewById(R.id.progress_overlay);
        progressWheel = findViewById(R.id.progressBar1);
        progressWheel.setVisibility(View.GONE);

        //mButtonMNIST = findViewById(R.id.mnistbutton);
        mButtonUpload = findViewById(R.id.uploadbutton);
        mButtonSave = findViewById(R.id.savebutton);
        mButtonDefocus = findViewById(R.id.defocusbutton);
        mButtonDeblur = findViewById(R.id.deblurbutton);
        mImageView = findViewById(R.id.imageview);
        mdefocusSwitch = findViewById(R.id.defocusSwitch);
        mrefocusSwitch = findViewById(R.id.refocusSwitch);

        mdefocusSwitch.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(mImageView.getDrawable()==null)
                {
                    Toast.makeText(MainActivity.this, "Image not Loaded. Please load it from Gallery",
                            Toast.LENGTH_SHORT).show();
                }
                else {
                    Bitmap inp_image1 = ((BitmapDrawable) mImageView.getDrawable()).getBitmap();
                    mImageView.setImageBitmap(inp_image1);
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                runOnUiThread(
                                        new Runnable() {
                                            @Override
                                            public void run() {
                                                progressWheel.setVisibility(View.VISIBLE);
                                            }
                                        });
                                runDefocusModel();
                                runOnUiThread(
                                        new Runnable() {
                                            @Override
                                            public void run() {
                                                progressWheel.setVisibility(View.GONE);
                                            }
                                        });
                            } catch (final Exception e) {
                                //if they aren't found, throw an error!
                                throw new RuntimeException("Error running defocus!", e);
                            }
                        }
                    }).start();
                }
            }
        });

        mrefocusSwitch.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(mImageView.getDrawable()==null)
                {
                    Toast.makeText(MainActivity.this, "Image not Loaded. Please load it from Gallery",
                            Toast.LENGTH_SHORT).show();
                }
                else {
                    Bitmap inp_image1 = ((BitmapDrawable) mImageView.getDrawable()).getBitmap();
                    mImageView.setImageBitmap(inp_image1);
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                runOnUiThread(
                                        new Runnable() {
                                            @Override
                                            public void run() {
                                                progressWheel.setVisibility(View.VISIBLE);
                                            }
                                        });
                                runRefocusModel();
                                runOnUiThread(
                                        new Runnable() {
                                            @Override
                                            public void run() {
                                                progressWheel.setVisibility(View.GONE);
                                            }
                                        });
                            } catch (final Exception e) {
                                //if they aren't found, throw an error!
                                throw new RuntimeException("Error running defocus!", e);
                            }
                        }
                    }).start();
                }
            }
        });

        mButtonUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, RESULT_LOAD_IMAGE);
            }
        });

        mButtonSave.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Bitmap saveBitmap = ((BitmapDrawable)mImageView.getDrawable()).getBitmap();
                String savedImageURL = MediaStore.Images.Media.insertImage(getContentResolver(), saveBitmap, "img", "img of output");
                Toast.makeText(MainActivity.this, "image saved to Gallery",
                        Toast.LENGTH_SHORT).show();
            }
        });

//        mButtonDefocus.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                Bitmap inp_image1 = ((BitmapDrawable)mImageView.getDrawable()).getBitmap();
//                mImageView.setImageBitmap(inp_image1);
////                Thread thread = new Thread(MainActivity.this);
////                thread.start();
//
//                //progressWheel.setVisibility(View.VISIBLE);
//                new Thread(new Runnable() {
//                    @Override
//                    public void run() {
//                        try {
//                            runOnUiThread(
//                                    new Runnable() {
//                                        @Override
//                                        public void run() {
//                                            progressWheel.setVisibility(View.VISIBLE);
//                                        }
//                                    });
//                            runDefocusModel();
//                            runOnUiThread(
//                                    new Runnable() {
//                                        @Override
//                                        public void run() {
//                                            progressWheel.setVisibility(View.GONE);
//                                        }
//                                    });
//                        } catch (final Exception e) {
//                            //if they aren't found, throw an error!
//                            throw new RuntimeException("Error running deblur!", e);
//                        }
//                    }
//                }).start();
//                //progressWheel.setVisibility(View.GONE);
//            }
//        });

        mButtonDeblur.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(mImageView.getDrawable()==null)
                {
                    Toast.makeText(MainActivity.this, "Image not Loaded. Please load it from Gallery",
                            Toast.LENGTH_SHORT).show();
                }
                else {
                    Bitmap inp_image1 = ((BitmapDrawable) mImageView.getDrawable()).getBitmap();
                    mImageView.setImageBitmap(inp_image1);
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                runOnUiThread(
                                        new Runnable() {
                                            @Override
                                            public void run() {
                                                progressWheel.setVisibility(View.VISIBLE);
                                            }
                                        });
                                runDeblurModel();
                                runOnUiThread(
                                        new Runnable() {
                                            @Override
                                            public void run() {
                                                progressWheel.setVisibility(View.GONE);
                                            }
                                        });
                            } catch (final Exception e) {
                                //if they aren't found, throw an error!
                                throw new RuntimeException("Error running defocus!", e);
                            }
                        }
                    }).start();
                }
            }
        });
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("Hello", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mOpenCVCallBack);
        } else {
            Log.d("Hello", "OpenCV library found inside package. Using it!");
            mOpenCVCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==RESULT_LOAD_IMAGE && resultCode == RESULT_OK && data!=null)
        {
            Uri selectedImage = data.getData();
            try{
                bitmap_orig = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                Bitmap scaledPhoto_orig = Bitmap.createScaledBitmap(bitmap_orig, WANTED_WIDTH, WANTED_HEIGHT, true);
                mImageView.setImageBitmap(scaledPhoto_orig);

            } catch (IOException ie){
                ie.printStackTrace();
            }
        }
    }

//    @Override
//    public void run() {
//        runPix2PixBlurryModel();
//    }

    void runDefocusModel() {
        int[] intValues = new int[WANTED_WIDTH * WANTED_HEIGHT];
        int[] focusValues = new int[WANTED_WIDTH * WANTED_WIDTH];
        float[] fmeasure = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];
        float[] floatValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 4];
        float[] outputValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];
        float[] mag_Values = new float[1];
//        if(mdefocusSwitch.getText()=="1")
//            mag_Values[0] = 3.0f;
//        else
//            mag_Values[0] = 0.0f;
        mag_Values[0] = 3.0f;

        try {
            //Bitmap bitmap = ((BitmapDrawable)mImageView.getDrawable()).getBitmap();
            if(mdefocusSwitch.isChecked()) {
                Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap_orig, WANTED_WIDTH, WANTED_HEIGHT, true);
                scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());

                //            //Bitmap bitmap_f = ((BitmapDrawable)mImageView_f.getDrawable()).getBitmap();
                //            Bitmap scaledBitmap_f = Bitmap.createScaledBitmap(mFocusMeasure, WANTED_WIDTH, WANTED_HEIGHT, true);
                //            scaledBitmap_f.getPixels(focusValues, 0, scaledBitmap_f.getWidth(), 0, 0, scaledBitmap_f.getWidth(), scaledBitmap_f.getHeight());

                //Bitmap inp_image = ((BitmapDrawable)mImageView.getDrawable()).getBitmap();
                Mat img = new Mat(bitmap_orig.getWidth(), bitmap_orig.getHeight(), CvType.CV_8UC1);
                Mat focus = new Mat(bitmap_orig.getWidth(), bitmap_orig.getHeight(), CvType.CV_8UC1);
                Utils.bitmapToMat(bitmap_orig, img);
                Imgproc.Laplacian(img, focus, CvType.CV_8UC1);
                Bitmap bitmap_f = scaledBitmap.copy(scaledBitmap.getConfig(), true);
                Utils.matToBitmap(focus, bitmap_f);

                Bitmap scaledBitmap_f = Bitmap.createScaledBitmap(bitmap_f, WANTED_WIDTH, WANTED_HEIGHT, true);
                scaledBitmap_f.getPixels(focusValues, 0, scaledBitmap_f.getWidth(), 0, 0, scaledBitmap_f.getWidth(), scaledBitmap_f.getHeight());


                for (int i = 0; i < focusValues.length; ++i) {
                    final int val1 = focusValues[i];
                    fmeasure[i * 3] = (((val1 >> 16) & 0xFF) - 128.0f) / 128.0f;
                    fmeasure[i * 3 + 1] = (((val1 >> 8) & 0xFF) - 128.0f) / 128.0f;
                    fmeasure[i * 3 + 2] = ((val1 & 0xFF) - 128.0f) / 128.0f;
                }

                for (int i = 0; i < intValues.length; ++i) {
                    final int val = intValues[i];
                    floatValues[i * 4] = (((val >> 16) & 0xFF) - 128.0f) / 128.0f;
                    floatValues[i * 4 + 1] = (((val >> 8) & 0xFF) - 128.0f) / 128.0f;
                    floatValues[i * 4 + 2] = ((val & 0xFF) - 128.0f) / 128.0f;
                    floatValues[i * 4 + 3] = fmeasure[i * 3];
                }
                AssetManager assetManager = getAssets();
                mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE2);
                mInferenceInterface.feed(INPUT_NODE, floatValues, 1, WANTED_HEIGHT, WANTED_WIDTH, 4);
                mInferenceInterface.feed(MAG_NODE, mag_Values, 1, 1);
                mInferenceInterface.run(new String[]{OUTPUT_NODE}, false);
                mInferenceInterface.fetch(OUTPUT_NODE, outputValues);

                for (int i = 0; i < intValues.length; ++i) {
                    intValues[i] = 0xFF000000
                            | (((int) ((outputValues[i * 3] + 1) * 127.5f)) << 16)
                            | (((int) ((outputValues[i * 3 + 1] + 1) * 127.5f)) << 8)
                            | ((int) ((outputValues[i * 3 + 2] + 1) * 127.5f));
                }

                Bitmap outputBitmap = scaledBitmap.copy(scaledBitmap.getConfig(), true);
                outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
                mGeneratedBitmap = Bitmap.createScaledBitmap(outputBitmap, bitmap_orig.getWidth(), bitmap_orig.getHeight(), true);
            }
            else
            {
                mGeneratedBitmap = Bitmap.createScaledBitmap(bitmap_orig, bitmap_orig.getWidth(), bitmap_orig.getHeight(), true);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        mImageView.setImageBitmap(mGeneratedBitmap);
                    }
                });
    }

    void runDeblurModel() {
        int[] intValues = new int[WANTED_WIDTH * WANTED_HEIGHT];
        float[] floatValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];
        float[] outputValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];

        try {
            Bitmap bitmap = ((BitmapDrawable)mImageView.getDrawable()).getBitmap();
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, WANTED_WIDTH, WANTED_HEIGHT, true);
            scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());

            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues[i * 3] = (((val >> 16) & 0xFF) - 128.0f) / 128.0f;
                floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - 128.0f) / 128.0f;
                floatValues[i * 3 + 2] = ((val & 0xFF) - 128.0f) / 128.0f;
            }

            AssetManager assetManager = getAssets();
            mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE1);
            mInferenceInterface.feed(INPUT_NODE, floatValues, 1, WANTED_HEIGHT, WANTED_WIDTH, 3);
            mInferenceInterface.run(new String[]{OUTPUT_NODE}, false);
            mInferenceInterface.fetch(OUTPUT_NODE, outputValues);

            for (int i = 0; i < intValues.length; ++i) {
                intValues[i] = 0xFF000000
                        | (((int) ((outputValues[i * 3]+1) * 127.5f)) << 16)
                        | (((int) ((outputValues[i * 3 + 1]+1) * 127.5f)) << 8)
                        | ((int) ((outputValues[i * 3 + 2]+1) * 127.5f));
            }

            Bitmap outputBitmap = scaledBitmap.copy(scaledBitmap.getConfig(), true);
            outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
            mGeneratedBitmap = Bitmap.createScaledBitmap(outputBitmap, bitmap.getWidth(), bitmap.getHeight(), true);

        } catch (Exception e) {
            e.printStackTrace();
        }
        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        mImageView.setImageBitmap(mGeneratedBitmap);
                    }
                });
    }

    //Below is the code for Refocus mdel
    void runRefocusModel() {
        int[] intValues = new int[WANTED_WIDTH * WANTED_HEIGHT];
        int[] focusValues = new int[WANTED_WIDTH * WANTED_WIDTH];
        float[] fmeasure = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];
        float[] floatValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 4];
        float[] outputValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];

        float[] floatValues1 = new float[WANTED_WIDTH * WANTED_HEIGHT * 6];
        float[] refocus_parameters = new float[2];
        if(mrefocusSwitch.isChecked()) {
            //convert from near to far
            refocus_parameters[1] = 8.0f;
            refocus_parameters[0] = 0.0f;
        }
        else {
            //convert from far to near
            refocus_parameters[0] = -8.0f;
            refocus_parameters[1] = 0.0f;
        }

        try {
            Bitmap bitmap = ((BitmapDrawable)mImageView.getDrawable()).getBitmap();
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, WANTED_WIDTH, WANTED_HEIGHT, true);
            scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());

            //Bitmap bitmap_f = ((BitmapDrawable)mImageView_f.getDrawable()).getBitmap();
            Bitmap scaledBitmap_f = Bitmap.createScaledBitmap(mFocusMeasure, WANTED_WIDTH, WANTED_HEIGHT, true);
            scaledBitmap_f.getPixels(focusValues, 0, scaledBitmap_f.getWidth(), 0, 0, scaledBitmap_f.getWidth(), scaledBitmap_f.getHeight());

            for(int i=0;i<focusValues.length;++i) {
                final int val1 = focusValues[i];
                fmeasure[i * 3] = (((val1 >> 16) & 0xFF) - 128.0f) / 128.0f;
                fmeasure[i * 3 + 1] = (((val1 >> 8) & 0xFF) - 128.0f) / 128.0f;
                fmeasure[i * 3 + 2] = ((val1 & 0xFF) - 128.0f) / 128.0f;
            }

            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues[i * 4] = (((val >> 16) & 0xFF) - 128.0f) / 128.0f;
                floatValues[i * 4 + 1] = (((val >> 8) & 0xFF) - 128.0f) / 128.0f;
                floatValues[i * 4 + 2] = ((val & 0xFF) - 128.0f) / 128.0f;
                floatValues[i * 4 + 3] = fmeasure[i * 3];
            }
            AssetManager assetManager = getAssets();
            mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE3);
            mInferenceInterface.feed(INPUT_NODE, floatValues, 1, WANTED_HEIGHT, WANTED_WIDTH, 4);
            mInferenceInterface.run(new String[]{OUTPUT_NODE}, false);
            mInferenceInterface.fetch(OUTPUT_NODE, outputValues);

            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues1[i * 6] = (((val >> 16) & 0xFF) - 128.0f) / 128.0f;
                floatValues1[i * 6 + 1] = (((val >> 8) & 0xFF) - 128.0f) / 128.0f;
                floatValues1[i * 6 + 2] = ((val & 0xFF) - 128.0f) / 128.0f;
                floatValues1[i * 6 + 3] = outputValues[i * 3];
                floatValues1[i * 6 + 4] = outputValues[i * 3 + 1];
                floatValues1[i * 6 + 5] = outputValues[i * 3 + 2];
            }

            mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE4);
            mInferenceInterface.feed(INPUT_NODE, floatValues1, 1, WANTED_HEIGHT, WANTED_WIDTH, 6);
            mInferenceInterface.feed(DELTA_NODE, refocus_parameters, 1, 2);
            mInferenceInterface.run(new String[]{OUTPUT_NODE}, false);
            mInferenceInterface.fetch(OUTPUT_NODE, outputValues);

            for (int i = 0; i < intValues.length; ++i) {
                intValues[i] = 0xFF000000
                        | (((int) ((outputValues[i * 3]+1) * 127.5f)) << 16)
                        | (((int) ((outputValues[i * 3 + 1]+1) * 127.5f)) << 8)
                        | ((int) ((outputValues[i * 3 + 2]+1) * 127.5f));
            }

            Bitmap outputBitmap = scaledBitmap.copy(scaledBitmap.getConfig(), true);
            outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
            mGeneratedBitmap = Bitmap.createScaledBitmap(outputBitmap, bitmap.getWidth(), bitmap.getHeight(), true);

        } catch (Exception e) {
            e.printStackTrace();
        }
        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        mImageView.setImageBitmap(mGeneratedBitmap);
                    }
                });
    }
}

