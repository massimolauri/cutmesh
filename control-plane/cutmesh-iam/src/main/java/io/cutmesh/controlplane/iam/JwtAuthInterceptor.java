package io.cutmesh.controlplane.iam;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

/**
 * Enforces HMAC-SHA256 JWT authentication on all Control Plane management endpoints (`/api/v1/**`),
 * excluding public login/register endpoints and internal GPU worker heartbeat routes.
 */
@Component
public class JwtAuthInterceptor implements HandlerInterceptor {

    public static final String AUTHENTICATED_USER_ID_ATTR = "authenticatedUserId";

    private final JwtTokenService jwtTokenService;

    public JwtAuthInterceptor(JwtTokenService jwtTokenService) {
        this.jwtTokenService = jwtTokenService;
    }

    @Override
    public boolean preHandle(
        HttpServletRequest request,
        HttpServletResponse response,
        Object handler
    ) {
        if ("OPTIONS".equalsIgnoreCase(request.getMethod())) {
            return true;
        }
        String authHeader = request.getHeader("Authorization");
        String userId = jwtTokenService.verifyAndExtractUserId(authHeader);
        request.setAttribute(AUTHENTICATED_USER_ID_ATTR, userId);
        return true;
    }
}

