package com.example.reconocimientodeimagenes;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Toast;

import com.example.reconocimientodeimagenes.classifier.ClasificadorDeImagen;

import org.checkerframework.common.subtyping.qual.Bottom;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    /**
     *Códigos de solicitudes para identificar solicitudes de permisos y cámaras
     */
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1000;
    private static final int CAMERA_REQEUST_CODE = 10001;
    /**
     * Elementos UI
     */
    private ImageView imageView;
    private ListView listView;
    private ClasificadorDeImagen imageClassifier;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // initalizing ui elements
        initializeUIElements();
    }
    /**
     * Método para inicializar elementos de la interfaz de usuario. este método agrega el clic
     */
    private void initializeUIElements() {
        imageView = findViewById(R.id.imagen_tomada);
        listView = findViewById(R.id.lv_probabilidades);
        Button takepicture = findViewById(R.id.btn_tomarFoto);
        /*
         * Creando una instancia de nuestro clasificador de imágenes de tensor
         */
        try {
            imageClassifier = new ClasificadorDeImagen(this);
        } catch (IOException e) {
            Log.e("Image Classifier Error", "ERROR: " + e);
        }
        // agregando el listener de clic al botón
        takepicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // comprobar si los permisos de la cámara están disponibles.
                //si el permiso está disponible, abrimos la intención de la
                // cámara para obtener una imagen
                // de lo contrario, solicitudes de permisos
                if (hasPermission()) {
                    openCamera();
                } else {
                    requestPermission();
                }
            }
        });
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        // si este es el resultado de nuestra solicitud de imagen de cámara
        if (requestCode == CAMERA_REQEUST_CODE) {
            // obteniendo mapa de bits de la imagen
            Bitmap photo = (Bitmap) Objects.requireNonNull(Objects.requireNonNull(data).getExtras()).get("data");
            // mostrando este mapa de bits en la vista de imagen
            imageView.setImageBitmap(photo);
            // pasar este mapa de bits al clasificador para hacer predicciones
            List<ClasificadorDeImagen.Recognition> predicitons = imageClassifier.recognizeImage(
                    photo, 0);
            //creando una lista de cadenas para mostrar en la vista de lista
            final List<String> predicitonsList = new ArrayList<>();
            for (ClasificadorDeImagen.Recognition recog : predicitons) {
                predicitonsList.add(recog.getName() + "  ::::::::::  " + recog.getConfidence());
            }
            // crear un adaptador de matriz para mostrar el resultado de la clasificación en la vista de lista
            ArrayAdapter<String> predictionsAdapter = new ArrayAdapter<>(
                    this, R.layout.support_simple_spinner_dropdown_item, predicitonsList);
            listView.setAdapter(predictionsAdapter);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // si este es el resultado de nuestra solicitud de permiso de cámara
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (hasAllPermissions(grantResults)) {
                openCamera();
            } else {
                requestPermission();
            }
        }
    }

    private boolean hasAllPermissions(int[] grantResults) {
        for (int result : grantResults) {
            if (result == PackageManager.PERMISSION_DENIED)
                return false;
        }
        return true;
    }
    /**
     * Solicitudes de método de permiso si la versión de Android es marshmallow o superior
     */
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            // si se puede solicitar permiso o no
            if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
                Toast.makeText(this, "Camera Permission Required", Toast.LENGTH_SHORT).show();
            }
            // solicitar el permiso de la cámara
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        }
    }
    /**
     * crea e inicia la intención de obtener una imagen de la cámara
     */
    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, CAMERA_REQEUST_CODE);
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        }
        return true;
    }
}