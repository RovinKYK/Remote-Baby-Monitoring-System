package com.example.baby_monitor_app;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.ImageView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    String[] modes = {"Sleeping", "Awaken", "Crying", "Disconnected"};
    String[] toast_msgs = {"Monitoring till baby awakes", "Monitors till baby cries", "Baby is crying!", "Device disconnected"};
    int[] img_Src = {R.drawable.babymonitoring, R.drawable.babymonitoring, R.drawable.backai, R.drawable.poweroff};

    ImageView main_img;
    AutoCompleteTextView autoCompleteTextView;
    ArrayAdapter<String> adapter;

    protected void change_mode(int mode) {
        Toast.makeText(MainActivity.this, toast_msgs[mode], Toast.LENGTH_SHORT).show();
        main_img.setImageResource(img_Src[mode]);

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        main_img = findViewById(R.id.mainImg);

        autoCompleteTextView = findViewById(R.id.autoComplete);
        adapter = new ArrayAdapter<String>(this,R.layout.list_item, modes);
        autoCompleteTextView.setAdapter(adapter);
        autoCompleteTextView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                change_mode(position);
            }
        });
    }
}