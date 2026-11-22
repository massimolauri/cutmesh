package io.cutmesh.controlplane;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class CutmeshControlPlaneApplication {

    public static void main(String[] args) {
        SpringApplication.run(CutmeshControlPlaneApplication.class, args);
    }
}
