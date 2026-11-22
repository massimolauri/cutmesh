package io.cutmesh.controlplane.iam;

import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface AppUserRepository extends JpaRepository<AppUser, String> {
    Optional<AppUser> findByEmailIgnoreCase(String email);
    Optional<AppUser> findByOauthProviderAndOauthSubject(String oauthProvider, String oauthSubject);
}

