package io.cutmesh.controlplane.shared;

import io.cutmesh.controlplane.iam.JwtAuthInterceptor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig {

    private final JwtAuthInterceptor jwtAuthInterceptor;

    public CorsConfig(JwtAuthInterceptor jwtAuthInterceptor) {
        this.jwtAuthInterceptor = jwtAuthInterceptor;
    }

    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/**")
                        .allowedOriginPatterns("*")
                        .allowedMethods("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
                        .allowedHeaders("*")
                        .allowCredentials(true);
            }

            @Override
            public void addInterceptors(InterceptorRegistry registry) {
                registry.addInterceptor(jwtAuthInterceptor)
                        .addPathPatterns("/api/v1/**")
                        .excludePathPatterns(
                                "/api/v1/public/status",
                                "/api/v1/auth/login",
                                "/api/v1/auth/register",
                                "/api/v1/auth/google",
                                "/api/v1/auth/api-key",
                                "/api/v1/internal/**"
                        );
            }
        };
    }
}
